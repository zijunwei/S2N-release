# evaluate based on presegmented shots
# load features for each presegmented shots
# the scores for each segment is calculated based on the average 
# get the length of each segment (useful for DP)
# use c3d features

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

    def __init__(self, dataset_name='TVSum', split='train', max_input_len=130, maximum_outputs=26,
                 feature_file_ext='npy', sum_budget=0.15, train_val_perms=None, eval_metrics='avg', data_path=None):

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
        self.filenames = [self.filenames[i] for i in train_val_perms]

        self.max_input_len = max_input_len
        self.maximum_outputs = maximum_outputs

        self.segment_features = []
        self.groundtruthscores = []
        self.video_nfps = []
        self.video_n_frames = []
        self.video_cps = []
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

            # load raw scores
            # # the size of raw_user_summaris is [len, num_users]
            # raw_user_summaris = np.array(raw_annotation_data[s_filename]).transpose()

            # change_points, each segment is [cps[i,0]: cps[i,1]+1]
            cps = Kydataset[KyKey]['change_points'][...]
            # total number of frames
            num_frames = Kydataset[KyKey]['n_frames'][()]
            assert n_frames == num_frames, 'frame length should be the same'
            # num of frames per segment
            nfps = Kydataset[KyKey]['n_frame_per_seg'][...]

            num_segments = cps.shape[0]

            # the size of s_features is: [length, fea_dim]
            s_video_features = np.load(
                os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_filename, self.feature_file_ext)))
            s_features_len = len(s_video_features)
            # the length of c3d feature is larger than annotation, choose middles to match
            assert abs(n_frames - s_features_len) < 6, 'annotation and feature length not equal! {:d}, {:d}'.format(
                n_frames, s_features_len)
            offset = abs(s_features_len - n_frames) / 2
            s_video_features = s_video_features[offset:offset + n_frames]

            # get average features
            s_segment_features = LoaderUtils.get_avg_seg_features(s_video_features, cps, num_segments)

            self.groundtruthscores.append(s_scores)
            self.segment_features.append(s_segment_features)
            self.videonames.append(s_filename)
            self.video_cps.append(cps)
            self.video_nfps.append(nfps)
            self.video_n_frames.append(n_frames)

        self.dataset_size = len(self.segment_features)
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

        for video_idx, (s_name, s_feature, s_groundtruth, s_cps, s_nfps, s_n_frames) in enumerate(zip(self.videonames, self.segment_features, self.groundtruthscores, self.video_cps, self.video_nfps, self.video_n_frames)):
            # pad s_feature to max_input_len
            [s_n_segment, fea_dim] = s_feature.shape
            s_feature_pad = np.zeros([self.max_input_len, fea_dim])
            s_feature_pad[:s_n_segment, :] = s_feature
            s_feature = s_feature_pad

            s_feature = Variable(torch.FloatTensor(s_feature), requires_grad=False)
            if use_cuda:
                s_feature = s_feature.cuda()

            s_feature = s_feature.permute(1,0).unsqueeze(0)

            pointer_probs, pointer_positions, cls_scores, _ = model(s_feature)

            cls_scores = cls_scores.contiguous().squeeze(2).squeeze(0)
            pred_scores = cls_scores.data.cpu().numpy()
            pointer_positions = pointer_positions.squeeze(0)
            pred_positions = pointer_positions.data.cpu().numpy()
            pred_positions = pred_positions.astype('int')

            pred_segment_scores = np.zeros([s_n_segment])
            pred_positions = pointer_positions.data.cpu().numpy()
            pred_segment_scores[pred_positions] = pred_scores

            pred_framescores = sum_tools.generate_summary(pred_segment_scores, s_cps, s_n_frames, list(s_nfps), proportion=0.15)
            s_F1, _, _ = sum_tools.evaluate_summary(pred_framescores, s_groundtruth.transpose(), self.eval_metrics)

            F1s += s_F1
            widgets[-6] = progressbar.FormatLabel('{:s}'.format(s_name))
            widgets[-4] = progressbar.FormatLabel('{:.4f}'.format(s_F1))
            pbar.update(video_idx)

        if n_notselected_seq > 0:
            print("not selected sequence:{:d}".format(n_notselected_seq))

        return F1s/self.dataset_size


if __name__ == '__main__':

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

