# evaluate based on ky annotations

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
import pickle as pkl
import random
# user_root = os.path.expanduser('~')
# KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')

# datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
# eval_metrics = {'summe': 'max', 'tvsum': 'max'}
# data_loaders = {'summe': SumMeLoader, 'tvsum':TVSumLoader}
class Evaluator():

    def __init__(self, dataset_name='TVSum', split='train', seq_length=90, overlap=0.9, sample_rate=None,
                 feature_file_ext='npy', sum_budget=0.15, train_val_perms=None, eval_metrics='max', data_path=None):

        if dataset_name.lower() not in ['summe', 'tvsum']:
            print('Unrecognized dataset {:s}'.format(dataset_name))
        self.dataset_name = dataset_name
        self.eval_metrics = eval_metrics#[self.dataset_name.lower()]
        self.split = split
        self.sum_budget = sum_budget
        self.feature_file_ext = feature_file_ext
        
        # self.feature_directory = os.path.join(user_root, 'datasets/%s/features/c3dd-red500' % (dataset_name))
        self.feature_directory = os.path.join(data_path, '%s/features/c3dd-red500' % (dataset_name))
        self.filenames = os.listdir(self.feature_directory)
        self.filenames = [f.split('.', 1)[0] for f in self.filenames]
        self.filenames.sort()
        n_files = len(self.filenames)
        # selected_perms = range(n_files)
        # if self.split == 'train':
        #     selected_perms = train_val_perms[:int(0.8 * n_files)]
        # elif self.split == 'val':
        #     selected_perms = train_val_perms[int(0.8 * n_files):]
        # else:
        #     print("Unrecognized split:{:s}".format(self.split))
        
        # self.filenames = [self.filenames[i] for i in selected_perms]
        self.filenames = [self.filenames[i] for i in train_val_perms]

        if sample_rate is None:
            self.sample_rate = [1, 2, 4]
        else:
            self.sample_rate = sample_rate
        self.seq_len = seq_length
        self.overlap = overlap

        self.videofeatures = []
        self.groundtruthscores = []
        self.combinegroundtruth01scores = []
        # self.segments = []
        self.videonames = []
        KY_dataset_path = os.path.join(data_path, 'KY_AAAI18/datasets')
        Kydataset = KyLoader.loadKyDataset(self.dataset_name.lower(), file_path=os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(dataset_name.lower())))
        conversion = KyLoader.loadConversion(self.dataset_name.lower(), file_path=os.path.join(KY_dataset_path, '{:s}_name_conversion.pkl'.format(dataset_name.lower())))
        self.raw2Ky = conversion[0]
        self.Ky2raw = conversion[1]

        # project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # raw_data_path = os.path.join(project_root, 'Devs_vsSum/datasets/TVSum/TVSumRaw.pkl')
        # raw_annotation_data = pkl.load(open(raw_data_path, 'rb'))

        for s_video_idx, s_filename in enumerate(self.filenames):
            KyKey = self.raw2Ky[s_filename]

            s_scores = Kydataset[KyKey]['user_summary'][...]
            s_scores = s_scores.transpose()
            n_frames = len(s_scores)
            # s_segments = LoaderUtils.convertlabels2segs(s_scores)

            # raw_user_summaris = raw_annotation_data[s_filename]
            # raw_user_summaris_01 = []
            # for s_raw_user_summary in raw_user_summaris:
            #     assert len(s_raw_user_summary) == n_frames

            #     s_raw_user_summary = np.expand_dims(np.array(s_raw_user_summary), -1)
            #     s_summary_segments, s_summary_scores = LoaderUtils.convertscores2segs(s_raw_user_summary)
            #     s_selected_segments = rep_conversions.selecteTopSegments(s_summary_segments, s_summary_scores, n_frames)
            #     # raw_user_summaris_01.append(s_segments)
            #     s_frame01scores = rep_conversions.keyshots2frame01scores(s_selected_segments, n_frames)
            #     raw_user_summaris_01.append(s_frame01scores)
            # raw_user_summaris_01 = np.stack(raw_user_summaris_01, axis=1)


            # raw_user_summaris = np.array(raw_user_summaris)
            # raw_user_summaris = raw_user_summaris.transpose()
            ky_combine_summaris = np.mean(s_scores, 1, keepdims=True)
            s_combine_segments, s_combine_segment_scores = LoaderUtils.convertscores2segs(ky_combine_summaris)
            s_combine_selected_segments = rep_conversions.selecteTopSegments(s_combine_segments, s_combine_segment_scores, n_frames)
            s_combine_frame01scores = rep_conversions.keyshots2frame01scores(s_combine_selected_segments, n_frames)


            # the size of s_features is: [length, fea_dim]
            s_video_features = np.load(
                os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_filename, self.feature_file_ext)))
            s_features_len = len(s_video_features)
            # the length of c3d feature is larger than annotation, choose middles to match
            assert abs(n_frames - s_features_len) < 6, 'annotation and feature length not equal! {:d}, {:d}'.format(
                n_frames, s_features_len)
            offset = abs(s_features_len - n_frames) / 2
            s_video_features = s_video_features[offset:offset + n_frames]

            self.groundtruthscores.append(s_scores)
            self.videofeatures.append(s_video_features)
            self.combinegroundtruth01scores.append(s_combine_frame01scores)
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
        for video_idx, (s_name, s_feature, s_groundtruth) in enumerate(zip(self.videonames, self.videofeatures, self.groundtruthscores)):
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
                        cls_scores = F.softmax(cls_scores, dim=-1)[:, -1]
                        
                        pred_segments.append(pred_positions.data.cpu().numpy())
                        pred_scores.append(cls_scores.data.cpu().numpy())

            
            #FIXME: debug here!
            pred_segments = np.concatenate(pred_segments)
            pred_scores = np.concatenate(pred_scores)
            updated_segments, updated_scores, picks = NMS.non_maxima_supression(pred_segments, pred_scores)
            selected_segments = rep_conversions.selecteTopSegments(updated_segments, updated_scores, n_frames)
            pred_framescores = rep_conversions.keyshots2frame01scores(selected_segments, n_frames)

            s_F1, _, _ = sum_tools.evaluate_summary(pred_framescores, s_groundtruth.transpose(), self.eval_metrics)

            F1s += s_F1
            widgets[-6] = progressbar.FormatLabel('{:s}'.format(s_name))
            widgets[-4] = progressbar.FormatLabel('{:.4f}'.format(s_F1))
            pbar.update(video_idx)

        if n_notselected_seq > 0:
            print("not selected sequence:{:d}".format(n_notselected_seq))

        return F1s/self.dataset_size


    # for each video, take the averaged annotation, use DP get a new ground truth, evaluate
    def Evaluate_upperbound_avg(self):
        F1s = 0
        n_notselected_seq = 0
        widgets = [' -- [ ',progressbar.Counter(), '|', str(self.dataset_size), ' ] ',
               progressbar.Bar(),  ' name:  ', progressbar.FormatLabel(''),
               ' F1s: ', progressbar.FormatLabel(''),
               ' (', progressbar.ETA(), ' ) ']

        pbar = progressbar.ProgressBar(max_value=self.dataset_size, widgets=widgets)
        pbar.start()

        #FIXME This process is problematic and needs update!
        for video_idx, (s_name, s_groundtruth, s_combine_01scores) in enumerate(zip(self.videonames, self.groundtruthscores, self.combinegroundtruth01scores)):
            n_frames = s_groundtruth.shape[0]

            s_F1, _, _ = sum_tools.evaluate_summary(s_combine_01scores, s_groundtruth.transpose(), self.eval_metrics)

            F1s += s_F1
            widgets[-6] = progressbar.FormatLabel('{:s}'.format(s_name))
            widgets[-4] = progressbar.FormatLabel('{:.4f}'.format(s_F1))
            pbar.update(video_idx)
            # print(s_F1)

        if n_notselected_seq > 0:
            print("not selected sequence:{:d}".format(n_notselected_seq))

        return F1s/self.dataset_size


    # for each video, random select a person, evaluate
    def Evaluate_upperbound_rdmuser(self):
        F1s = 0
        n_notselected_seq = 0
        widgets = [' -- [ ',progressbar.Counter(), '|', str(self.dataset_size), ' ] ',
               progressbar.Bar(),  ' name:  ', progressbar.FormatLabel(''),
               ' F1s: ', progressbar.FormatLabel(''),
               ' (', progressbar.ETA(), ' ) ']

        pbar = progressbar.ProgressBar(max_value=self.dataset_size, widgets=widgets)
        pbar.start()

        #FIXME This process is problematic and needs update!
        for video_idx, (s_name, s_groundtruth) in enumerate(zip(self.videonames, self.groundtruthscores)):
            n_frames = s_groundtruth.shape[0]

            n_users = s_groundtruth.shape[1]
            select_user_id = np.random.choice(n_users)
            pred_scores = s_groundtruth[:,select_user_id]

            s_F1, _, _ = sum_tools.evaluate_summary(pred_scores, s_groundtruth.transpose(), self.eval_metrics)

            F1s += s_F1
            widgets[-6] = progressbar.FormatLabel('{:s}'.format(s_name))
            widgets[-4] = progressbar.FormatLabel('{:.4f}'.format(s_F1))
            pbar.update(video_idx)
            # print(s_F1)

        if n_notselected_seq > 0:
            print("not selected sequence:{:d}".format(n_notselected_seq))

        return F1s/self.dataset_size

if __name__ == '__main__':

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

    kfold_split = 1
    train_val_perms = np.arange(50)
    random.Random(0).shuffle(train_val_perms)
    train_val_perms = train_val_perms.reshape([5, -1])
    train_perms = np.delete(train_val_perms, kfold_split, 0).reshape([-1])
    val_perms = train_val_perms[kfold_split]
    
    location = 'bigbrain'
    if location == 'home':
        data_path = os.path.join(os.path.expanduser('~'), 'datasets')
    else:
        data_path = os.path.join('/nfs/%s/boyu/SDN'%(location), 'datasets')
    
    val_evaluator = Evaluator(dataset_name='TVSum', split='val', seq_length=100, overlap=0.9, sample_rate=[16],
                sum_budget=0.15, train_val_perms=val_perms, eval_metrics='avg', data_path=data_path)
    
    print('avg')
    print(val_evaluator.Evaluate_upperbound_avg())

    print('random select one')
    f1s = 0
    for i in range(10):
        f1s += val_evaluator.Evaluate_upperbound_rdmuser()
    f1s = f1s/10
    print(f1s)


