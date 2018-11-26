import os
import numpy as np
import torch
from vsSummDevs.SumEvaluation import rep_conversions, NMS
from torch.autograd import Variable
from Losses import loss_transforms
import progressbar
import vsSummDevs.datasets.SumMe.BNInceptionFeatureLoader as SumMeLoader
import vsSummDevs.datasets.SumMe.path_vars as SumMe_paths
import vsSummDevs.datasets.TVSum.BNInceptionFeatureLoader as TVSumLoader
import vsSummDevs.datasets.TVSum.path_vars as TVSum_paths
from Devs_vsSum.datasets import LoaderUtils
from vsSummDevs.datasets import KyLoader
from vsSummDevs import JM as vsum_tools

user_root = os.path.expanduser('~')
KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')

datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
eval_metrics = {'summe': 'max', 'tvsum': 'avg'}
data_loaders = {'summe': SumMeLoader, 'tvsum':TVSumLoader}
class Evaluator():

    def __init__(self, dataset_name='SumMe', split='train', clip_size=100, sum_budget=0.15, sample_rate=None, overlap_ratio=0.25):

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
        self.overlap_ratio = overlap_ratio
        self.clip_size = clip_size
        if sample_rate is None:
            self.sample_rate = [1, 5, 10]
        else:
            if isinstance(sample_rate, list):
                self.sample_rate = sample_rate
            else:
                self.sample_rate = [sample_rate]

        self.videofeatures = []
        self.groundtruthscores = []
        self.segments = []
        self.videonames = []
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
            self.videonames.append(s_filename)
        self.dataset_size = len(self.videofeatures)
        print("{:s}\tEvaluator: {:s}\t{:d} Videos".format(self.dataset_name, self.split, self.dataset_size))


    def Evaluate(self, model, use_cuda=True):

        F1s = 0
        pbar = progressbar.ProgressBar(max_value=len(self.groundtruthscores))
        for video_idx, (s_name, s_feature, s_groundtruth, s_segments) in enumerate(zip(self.videonames, self.videofeatures, self.groundtruthscores, self.segments)):
            pbar.update(video_idx)
            n_frames = s_feature.shape[0]

            pred_segments = []
            pred_scores = []
            for s_sample_rate in self.sample_rate:
                sample_rate_feature = s_feature[::s_sample_rate, :]
                sample_rate_nframes = sample_rate_feature.shape[0]

                offset = int(self.overlap_ratio * self.clip_size)
                startingBounds = 0
                endingBounds = min(sample_rate_nframes, startingBounds + self.clip_size)
                proposedSegments = []
                while endingBounds < sample_rate_nframes:
                    proposedSegments.append([startingBounds, endingBounds])
                    startingBounds = max(0, endingBounds - offset)
                    endingBounds = startingBounds + self.clip_size

                # TODO Here could also be of change: record the clips and dynamic programming based on non-overlap segments and scores...
                for s_proposed_segment in proposedSegments:
                    startIdx = s_proposed_segment[0]
                    endIdx = s_proposed_segment[1]
                    if startIdx == endIdx:
                        continue
                    s_clip_feature = Variable(torch.FloatTensor(sample_rate_feature[startIdx:endIdx, :]),
                                              requires_grad=False)
                    if use_cuda:
                        s_clip_feature = s_clip_feature.cuda()

                    s_clip_feature = s_clip_feature.unsqueeze(0)
                    index_vector, segment_score = model(s_clip_feature)
                    pred_idxes = loss_transforms.torchVT_scores2indices(index_vector, topK=10)
                    real_startIdx = startIdx * s_sample_rate
                    pred_idxes *= s_sample_rate
                    if pred_idxes.data[0, 1, 0] <= pred_idxes.data[0, 0, 0]:
                        continue
                    pred_segments.append([real_startIdx + pred_idxes.data[0, 0, 0],
                                          min(real_startIdx + pred_idxes.data[0, 1, 0], n_frames)])
                    pred_scores.append(segment_score.data[0][0])

            pred_segments = np.asarray(pred_segments)
            pred_scores = np.asarray(pred_scores)
            updated_segments, updated_scores, picks = NMS.non_maxima_supression(pred_segments, pred_scores)
            selected_segments = rep_conversions.selecteTopSegments(updated_segments, updated_scores, n_frames)
            pred_framescores = rep_conversions.keyshots2frame01scores(selected_segments, n_frames)

            s_F1, _, _ = vsum_tools.evaluate_summary(pred_framescores, s_groundtruth.transpose(), self.eval_metrics)

            F1s += s_F1
            print ('{:s}\t{:.4f}'.format(s_name, s_F1))


        return F1s/self.dataset_size

if __name__ == '__main__':
    from PtrNet.PointerNet2PointDepOffset import PointerNet
    from PtUtils import cuda_model

    sDataset = Evaluator(dataset_name='SumMe',split='val', clip_size=100)
    model = PointerNet(input_dim=1024,
                       hidden_dim=512,
                       lstm_layers=1)
    model = cuda_model.convertModel2Cuda(model, gpu_id=1, multiGpu=False)
    maxF1 = sDataset.Evaluate(model, use_cuda=True)
    print("Randomly updated value: {:.4f}".format(maxF1))
    print("DEB")

