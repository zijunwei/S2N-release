import os
import numpy as np
import torch
from vsSummDevs.SumEvaluation import rep_conversions
from torch.autograd import Variable
from Losses import loss_transforms
import progressbar
import datasets.SumMe.BNInceptionFeatureLoader as SumMeLoader
import datasets.SumMe.path_vars as SumMe_paths
import datasets.TVSum.BNInceptionFeatureLoader as TVSumLoader
import datasets.TVSum.path_vars as TVSum_paths
from Devs_vsSum.datasets import LoaderUtils
from datasets import KyLoader
from vsSummDevs import JM as vsum_tools

user_root = os.path.expanduser('~')
KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')

datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
eval_metrics = {'summe': 'max', 'tvsum': 'avg'}
data_loaders = {'summe': SumMeLoader, 'tvsum':TVSumLoader}
class Evaluator():

    def __init__(self, dataset_name='SumMe', split='train', clip_size=50, sum_budget=0.15):

        if dataset_name.lower() not in ['summe', 'tvsum']:
            print('Unrecognized dataset {:s}'.format(dataset_name))
        self.dataset_name = dataset_name
        self.eval_metrics = eval_metrics[self.dataset_name.lower()]
        self.split = split
        self.sum_budget = sum_budget
        self.filenames = datasetpaths[self.dataset_name.lower()].file_names
        self.data_loader = data_loaders[self.dataset_name.lower()]

        n_files = len(self.filenames)

        if self.split == 'train':
            self.filenames = self.filenames[:int(0.8*n_files)]
        elif self.split == 'val':
            self.filenames = self.filenames[int(0.8*n_files):]

        else:
            print("Unrecognized split:{:s}".format(self.split))

        self.clip_size = clip_size
        self.videofeatures = []
        self.groundtruthscores = []
        self.segments = []

        Kydataset = KyLoader.loadKyDataset(self.dataset_name.lower())
        conversion = KyLoader.loadConversion(self.dataset_name.lower())
        self.raw2Ky = conversion[0]
        self.Ky2raw = conversion[1]


        for s_video_idx, s_filename in enumerate(self.filenames):
            KyKey = self.raw2Ky[s_filename]

            s_video_features, _, _  = self.data_loader.load_by_name(s_filename)
            s_scores = Kydataset[KyKey]['user_summary'][...]
            s_scores = s_scores.transpose()
            s_segments = LoaderUtils.convertlabels2segs(s_scores)

            self.groundtruthscores.append(s_scores)
            self.videofeatures.append(s_video_features)
            self.segments.append(s_segments)

        self.dataset_size = len(self.videofeatures)
        print("{:s}\tEvaluator: {:s}\t{:d} Videos".format(self.dataset_name, self.split, self.dataset_size))


    def EvaluateTopK(self, model, topK=1, use_cuda=True):

        F1s = 0
        pbar = progressbar.ProgressBar(max_value=len(self.groundtruthscores))
        for video_idx, (s_feature, s_groundtruth) in enumerate(zip(self.videofeatures, self.groundtruthscores, self.segments)):
            pbar.update(video_idx)
            n_frames = s_feature.shape[0]

            pred_vector = np.zeros(n_frames)
            n_clips = int(n_frames/self.clip_size) # TODO: remove the +1 to remove the last one ...
            boundaris = [0]
            for i in range(1, n_clips+1):
                boundaris.append(min(i*self.clip_size, n_frames))
            #TODO Here could also be of change: record the clips and dynamic programming based on non-overlap segments and scores...
            for clip_idx in range(n_clips):
                startIdx = boundaris[clip_idx]
                endIdx = boundaris[clip_idx+1]
                if startIdx == endIdx:
                    continue
                s_clip_feature = Variable(torch.FloatTensor(s_feature[startIdx:endIdx,:]), requires_grad=False)
                if use_cuda:
                    s_clip_feature = s_clip_feature.cuda()

                s_clip_feature = s_clip_feature.unsqueeze(0)
                index_vector, segment_score = model(s_clip_feature)
                pred_idxes = loss_transforms.torchVT_scores2indicesTopK(index_vector, topK=topK)

                for k in range(topK):
                    for i in range(startIdx+pred_idxes.data[0, 0, k], startIdx+pred_idxes.data[0, 1, k]):
                        pred_vector[i] = max(pred_vector[i], segment_score.data[0][0])

            pred_vector = rep_conversions.framescore2frame01score_sort(pred_vector)
            s_F1, _, _ = vsum_tools.evaluate_summary(pred_vector, s_groundtruth, self.eval_metrics)
            F1s += s_F1



        return F1s/self.dataset_size

if __name__ == '__main__':
    from PtrNet.PointerNet2PointDepOffset import PointerNet
    from PtUtils import cuda_model

    sDataset = Evaluator(dataset_name='SumMe',split='val', clip_size=100)
    model = PointerNet(input_dim=1024,
                       hidden_dim=512,
                       lstm_layers=1)
    model = cuda_model.convertModel2Cuda(model, gpu_id=1, multiGpu=False)
    maxF1 = sDataset.EvaluateTopK(model, use_cuda=True, topK=3)
    print("Randomly updated value: {:.4f}".format(maxF1))
    print("DEB")


