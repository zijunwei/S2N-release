# evaluator for tvsum
# load googlenet features, and segment level labels
# feature and labels per 15 frames
# load the averaged annotation for each video

import os
import numpy as np
import torch
from Devs_vsSum.SumEvaluation import rep_conversions, NMS
from torch.autograd import Variable
import progressbar
# import Devs_vsSum.datasets.SumMe.BNInceptionFeatureLoader as SumMeLoader
# import Devs_vsSum.datasets.SumMe.path_vars as SumMe_paths
# import Devs_vsSum.datasets.TVSum.BNInceptionFeatureLoader as TVSumLoader
# import Devs_vsSum.datasets.TVSum.path_vars as TVSum_paths
from Devs_vsSum.datasets import LoaderUtils
from Devs_vsSum.datasets import KyLoader
from Devs_vsSum.JM import sum_tools
import torch.nn.functional as F
from SDN import helper
# user_root = os.path.expanduser('~')
# KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')

# datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
# eval_metrics = {'summe': 'max', 'tvsum': 'max'}
# data_loaders = {'summe': SumMeLoader, 'tvsum':TVSumLoader}
class Evaluator():

    def __init__(self, dataset_name='TVSum', split='train', seq_length=90, overlap=0.9, sample_rate=None,
                 feature_file_ext='npy', sum_budget=0.15, train_val_perms=None, eval_metrics='avg', data_path=None):

        if dataset_name.lower() not in ['summe', 'tvsum']:
            print('Unrecognized dataset {:s}'.format(dataset_name))
        self.dataset_name = dataset_name
        self.eval_metrics = eval_metrics#[self.dataset_name.lower()]
        self.split = split
        self.sum_budget = sum_budget
        self.feature_file_ext = feature_file_ext
        
        self.feature_directory = os.path.join(data_path, '%s/features/c3dd-red500' % (dataset_name))
        self.filenames = os.listdir(self.feature_directory)
        self.filenames = [f.split('.', 1)[0] for f in self.filenames]
        self.filenames.sort()
        n_files = len(self.filenames)
        self.filenames = [self.filenames[i] for i in train_val_perms]

        if sample_rate is None:
            self.sample_rate = [1, 2, 4]
        else:
            self.sample_rate = sample_rate
        self.seq_len = seq_length
        self.overlap = overlap

        self.videofeatures = []
        self.groundtruthscores = []
        self.groundtruth01scores = []
        # self.segments = []
        self.videonames = []
        KY_dataset_path = os.path.join(data_path, 'KY_AAAI18/datasets')
        Kydataset = KyLoader.loadKyDataset(self.dataset_name.lower(), file_path=os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(dataset_name.lower())))
        conversion = KyLoader.loadConversion(self.dataset_name.lower(), file_path=os.path.join(KY_dataset_path, '{:s}_name_conversion.pkl'.format(dataset_name.lower())))
        self.raw2Ky = conversion[0]
        self.Ky2raw = conversion[1]

        for s_video_idx, s_filename in enumerate(self.filenames):
            KyKey = self.raw2Ky[s_filename]

            s_scores = Kydataset[KyKey]['gtscore'][...]
            s_scores = s_scores.reshape([-1, 1])

            n_frames = s_scores.shape[0]

            s_segments, s_segment_scores = LoaderUtils.convertscores2segs(s_scores)
            selected_segments = rep_conversions.selecteTopSegments(s_segments, s_segment_scores, n_frames)
            s_frame01scores = rep_conversions.keyshots2frame01scores(selected_segments, n_frames)
            # s_frame01scores = rep_conversions.framescore2frame01score_inteval(s_scores.reshape([-1]), s_segments, lratio=self.sum_budget)

            # the size of s_features is: [length, fea_dim]
            # s_video_features = np.load(
            #     os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_filename, self.feature_file_ext)))
            s_video_features = Kydataset[KyKey]['features']
            s_features_len = len(s_video_features)
            # the length of c3d feature is larger than annotation, choose middles to match
            assert abs(n_frames - s_features_len) < 6, 'annotation and feature length not equal! {:d}, {:d}'.format(
                n_frames, s_features_len)
            offset = abs(s_features_len - n_frames) / 2
            s_video_features = s_video_features[offset:offset + n_frames]

            self.groundtruthscores.append(s_scores)
            self.groundtruth01scores.append(s_frame01scores)
            self.videofeatures.append(s_video_features)
            # self.segments.append(s_segments)
            self.videonames.append(s_filename)
        self.dataset_size = len(self.videofeatures)
        print("{:s}\tEvaluator: {:s}\t{:d} Videos".format(self.dataset_name, self.split, self.dataset_size))


    def Evaluate(self, model, use_cuda=True):

        F1s = 0
        n_notselected_seq = 0
        widgets = [' -- [ ',progressbar.Counter(), '|', str(self.dataset_size), ' ] ',
               progressbar.Bar(),  ' name:  ', progressbar.FormatLabel(''),
               ' F1s: ', progressbar.FormatLabel(''),
               ' (', progressbar.ETA(), ' ) ']

        pbar = progressbar.ProgressBar(max_value=self.dataset_size, widgets=widgets)
        pbar.start()

        #FIXME This process is problematic and needs update!
        for video_idx, (s_name, s_feature, s_groundtruth01score) in enumerate(zip(self.videonames, self.videofeatures, self.groundtruth01scores)):
            n_frames = s_feature.shape[0]

            pred_segments = []
            pred_scores = []
            for s_sample_rate in self.sample_rate:
                sample_rate_feature = s_feature[::s_sample_rate, :]
                sample_rate_nframes = sample_rate_feature.shape[0]

                startingBounds = 0
                if sample_rate_nframes < self.seq_len:
                    n_notselected_seq += 1
                else:
                    isInbound = True
                    proposedSegments = []
                    while startingBounds < sample_rate_nframes and isInbound:
                        endingBounds = startingBounds + self.seq_len
                        if endingBounds >= sample_rate_nframes:
                            isInbound = False
                            endingBounds = sample_rate_nframes
                            startingBounds = endingBounds - self.seq_len
                        proposedSegments.append([startingBounds, endingBounds])
                        startingBounds += int((1-self.overlap)*self.seq_len)

                    # TODO Here could also be of change: record the clips and dynamic programming based on non-overlap segments and scores...
                    for s_proposed_segment in proposedSegments:
                        startIdx = s_proposed_segment[0]
                        endIdx = s_proposed_segment[1]
                        assert endIdx - startIdx == self.seq_len, "distance between startIdx and endIdx should be seq_len:{:d},{:d},{:d}".format(endIdx, startIdx, self.seq_len)
                        s_clip_feature = Variable(torch.FloatTensor(sample_rate_feature[startIdx:endIdx, :]),
                                                  requires_grad=False)
                        if use_cuda:
                            s_clip_feature = s_clip_feature.cuda()

                        s_clip_feature = s_clip_feature.permute(1,0).unsqueeze(0)

                        _, head_positions, _, tail_positions, cls_scores, _ = model(s_clip_feature)
                        head_positions, tail_positions = helper.switch_positions(head_positions, tail_positions)
                        # correct ones:
                        head_positions = (head_positions + startIdx) * s_sample_rate
                        tail_positions = (tail_positions + startIdx) * s_sample_rate

                        head_positions = head_positions.squeeze(0)
                        tail_positions = tail_positions.squeeze(0)
                        cls_scores = cls_scores.squeeze(0)

                        pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
                        # cls_scores = F.softmax(cls_scores, dim=-1)[:, -1]
                        cls_scores = F.hardtanh(cls_scores, min_val=0, max_val=1).contiguous().view(-1)
                        
                        pred_segments.append(pred_positions.data.cpu().numpy())
                        pred_scores.append(cls_scores.data.cpu().numpy())

            
            #FIXME: debug here!
            pred_segments = np.concatenate(pred_segments)
            pred_scores = np.concatenate(pred_scores)
            updated_segments, updated_scores, picks = NMS.non_maxima_supression(pred_segments, pred_scores)
            selected_segments = rep_conversions.selecteTopSegments(updated_segments, updated_scores, n_frames)
            pred_framescores = rep_conversions.keyshots2frame01scores(selected_segments, n_frames)

            s_F1, _, _ = sum_tools.evaluate_summary(pred_framescores, s_groundtruth01score.reshape([1, -1]), self.eval_metrics)

            F1s += s_F1
            widgets[-6] = progressbar.FormatLabel('{:s}'.format(s_name))
            widgets[-4] = progressbar.FormatLabel('{:.4f}'.format(s_F1))
            pbar.update(video_idx)

        if n_notselected_seq > 0:
            print("not selected sequence:{:d}".format(n_notselected_seq))

        return F1s/self.dataset_size

if __name__ == '__main__':
    pass

    # from PtrNet.PointerNet2PointDepOffset import PointerNet
    # from PtUtils import cuda_model
    #
    # sDataset = Evaluator(dataset_name='SumMe',split='val', clip_size=100)
    # model = PointerNet(input_dim=1024,
    #                    hidden_dim=512,
    #                    lstm_layers=1)
    # model = cuda_model.convertModel2Cuda(model, gpu_id=1, multiGpu=False)
    # maxF1 = sDataset.Evaluate(model, use_cuda=True)
    # print("Randomly updated value: {:.4f}".format(maxF1))
    # print("DEB")


