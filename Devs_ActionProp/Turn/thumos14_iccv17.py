import numpy as np
from math import sqrt
import os
import random
import pickle
import torch.utils.data
from torch.utils.data import DataLoader
import progressbar

def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return iou


def LoadFeatureSwin_Center(feature_directory, movie_name, start_id, end_id, unit_size=16., feature_size=2048):
    swin_step=unit_size
    all_feat=np.zeros([0, feature_size], dtype=np.float32)
    current_pos=start_id
    while current_pos<end_id:
        swin_start=current_pos
        swin_end=swin_start+swin_step
        feat=np.load(os.path.join(feature_directory, '{:s}.mp4_{:s}_{:s}.npy'.format(movie_name, str(swin_start), str(swin_end))))
        all_feat=np.vstack((all_feat,feat))
        current_pos += swin_step
    pool_feat=np.mean(all_feat,axis=0)
    return pool_feat


def LoadFeatureSwin_Left(feature_directory, movie_name, start_id, unit_size=16., feature_size=2048, n_ctx=4):
    swin_step=unit_size
    all_feat=np.zeros([0,feature_size],dtype=np.float32)
    count=0
    current_pos=start_id
    context_ext=False
    while count<n_ctx:
        swin_start=current_pos-swin_step
        swin_end=current_pos
        feature_path = os.path.join(feature_directory, '{:s}.mp4_{:s}_{:s}.npy'.format(movie_name, str(swin_start), str(swin_end)))
        if os.path.exists(feature_path):
            feat=np.load(feature_path)
            all_feat=np.vstack((all_feat,feat))
            context_ext=True
        current_pos-=swin_step
        count+=1
    if context_ext:
        pool_feat=np.mean(all_feat,axis=0)
    else:
        pool_feat=np.zeros([feature_size],dtype=np.float32)
    return np.reshape(pool_feat,[feature_size])


def LoadFeatureSwin_Right(feature_directory, movie_name, end_id, unit_size=16., feature_size=2048, n_ctx=4):
    swin_step=unit_size
    all_feat=np.zeros([0,feature_size],dtype=np.float32)
    count=0
    current_pos=end_id
    context_ext=False
    while count<n_ctx:
        swin_start=current_pos
        swin_end=current_pos+swin_step
        feature_path = os.path.join(feature_directory, '{:s}.mp4_{:s}_{:s}.npy'.format(movie_name, str(swin_start), str(swin_end)))
        if os.path.exists(feature_path):
            feat=np.load(feature_path)
            all_feat=np.vstack((all_feat,feat))
            context_ext=True
        current_pos+=swin_step
        count+=1
    if context_ext:
        pool_feat=np.mean(all_feat,axis=0)
    else:
        pool_feat=np.zeros([feature_size],dtype=np.float32)
    return np.reshape(pool_feat,[feature_size])


class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, feature_directory, foreground_path, background_path, n_ctx=4,
                 feature_size=2048, unit_size=16.):
        self.n_ctx = n_ctx
        self.unit_feature_size = feature_size
        self.unit_size = unit_size
        self.feature_directory = feature_directory
        self.instances = []

        print "Reading training data list from " + foreground_path + " and " + background_path
        with open(foreground_path) as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                gt_start = float(l.rstrip().split(" ")[3])
                gt_end = float(l.rstrip().split(" ")[4])
                round_gt_start = np.round(gt_start / unit_size) * self.unit_size + 1
                round_gt_end = np.round(gt_end / unit_size) * self.unit_size + 1
                self.instances.append(
                    (movie_name, clip_start, clip_end, gt_start, gt_end, round_gt_start, round_gt_end, 1))
        print str(len(self.instances)) + " Positive Samples"
        positive_num = len(self.instances) * 1.0
        with open(background_path) as f:
            for l in f:
                # control the ratio between  background samples and positive samples to be 10:1
                if random.random() > 10.0 * positive_num / 270000: continue
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                self.instances.append((movie_name, clip_start, clip_end, 0, 0, 0, 0, 0))
        self.num_instances = len(self.instances)
        print str(len(self.instances)) + " Total Samples"

    def rel_offset(self, clip_start, clip_end, round_gt_start, round_gt_end):
        start_offset = (round_gt_start - clip_start)*1. / self.unit_size
        end_offset = (round_gt_end - clip_end)*1. / self.unit_size
        return start_offset, end_offset

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):
        s_instance = self.instances[index]

        movie_name = s_instance[0]
        clip_start = s_instance[1]
        clip_end = s_instance[2]

        featmap = LoadFeatureSwin_Center(feature_directory=self.feature_directory, movie_name=movie_name,
                                         start_id=clip_start, end_id=clip_end,
                                         unit_size=self.unit_size, feature_size=self.unit_feature_size)
        left_feat = LoadFeatureSwin_Left(feature_directory=self.feature_directory, movie_name=movie_name,
                                         start_id=clip_start, n_ctx=self.n_ctx,
                                         unit_size=self.unit_size, feature_size=self.unit_feature_size)
        right_feat = LoadFeatureSwin_Right(feature_directory=self.feature_directory, movie_name=movie_name,
                                           end_id=clip_end, n_ctx=self.n_ctx,
                                           unit_size=self.unit_size, feature_size=self.unit_feature_size)
        s_feature = np.hstack((left_feat, featmap, right_feat))

        if s_instance[7]==1:
            round_gt_start = s_instance[5]
            round_gt_end = s_instance[6]
            s_label = [1]
            start_offset, end_offset = self.rel_offset(clip_start, clip_end, round_gt_start, round_gt_end)

            s_offset = [start_offset, end_offset]
            gt_clip = [round_gt_start, round_gt_end]
            # print str(clip_start)+" "+str(clip_end)+" "+str(round_gt_start)+" "+str(round_gt_end)+" "+str(start_offset)+" "+str(end_offset)
        else:
            s_label = [0]
            s_offset = [0, 0]
            gt_clip = [0, 0]

        s_feature = torch.FloatTensor(s_feature)
        s_label = torch.FloatTensor(s_label)
        s_offset = torch.FloatTensor(s_offset)
        gt_clip = torch.FloatTensor(gt_clip)
        return s_feature, s_offset, s_label, gt_clip


class EvaluateDataset(torch.utils.data.Dataset):
    def __init__(self, feature_directory, clip_path, n_ctx=4, feature_size=2048, unit_size=16.):
        self.feature_directory = feature_directory
        self.feature_size = feature_size
        self.instances = []
        self.n_ctx = n_ctx
        self.unit_size = unit_size
        self.movie_names = []
        with open(clip_path, 'rb') as f:
            for l in f:
                movie_name = l.rstrip().split(" ")[0]
                clip_start = float(l.rstrip().split(" ")[1])
                clip_end = float(l.rstrip().split(" ")[2])
                self.movie_names.append(movie_name)
                self.instances.append((movie_name, clip_start, clip_end))
        self.num_instances = len(self.instances)
        self.movie_names = list(set(self.movie_names))
        print "{:d} instances were used for Test".format(self.num_instances)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):
        s_instance = self.instances[index]

        movie_name = s_instance[0]
        movie_name_idx = self.movie_names.index(movie_name)
        clip_start = s_instance[1]
        clip_end = s_instance[2]
        featmap = LoadFeatureSwin_Center(feature_directory=self.feature_directory, movie_name=movie_name,
                                         start_id=clip_start, end_id=clip_end,
                                         unit_size=self.unit_size, feature_size=self.feature_size)

        left_feat = LoadFeatureSwin_Left(feature_directory=self.feature_directory, movie_name=movie_name,
                                         start_id=clip_start, n_ctx=self.n_ctx,
                                         unit_size=self.unit_size, feature_size=self.feature_size)

        right_feat = LoadFeatureSwin_Right(feature_directory=self.feature_directory, movie_name=movie_name,
                                           end_id=clip_end, n_ctx=self.n_ctx,
                                           unit_size=self.unit_size, feature_size=self.feature_size)
        feat = np.hstack((left_feat, featmap, right_feat))
        # feat = np.reshape(feat, [1, self.feature_size * 3])
        feat = torch.FloatTensor(feat)
        clip_position = torch.FloatTensor([clip_start, clip_end])
        movie_name_idx = torch.LongTensor([movie_name_idx])
        return feat, clip_position, movie_name_idx


        #TODO: pay attention here on the number of features needed!
        # outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
        # reg_end = clip_end + outputs[3] * unit_size
        # reg_start = clip_start + outputs[2] * unit_size
        # round_reg_end = clip_end + np.round(outputs[3]) * unit_size
        # round_reg_start = clip_start + np.round(outputs[2]) * unit_size
        # softmax_score = softmax(outputs[0:2])
        # action_score = softmax_score[1]
        # results_lst.append(
        #     (movie_name, round_reg_start, round_reg_end, reg_start, reg_end, action_score, outputs[0], outputs[1]))
        # pickle.dump(results_lst, open("./test_results/results_TURN_flow_iter" + str(iter_step) + ".pkl", "w"))



if __name__ == '__main__':
    batch_size = 12
    test_featmap_dir="/home/zwei/datasets/THUMOS14/features/DenseFlowICCV2017/test_fc6_16_overlap0.5_denseflow/"
    test_clip_path="/home/zwei/Dev/TURN_TAP_ICCV17/turn_codes/test_swin.txt"

    val_dataset = EvaluateDataset(feature_directory=test_featmap_dir, clip_path=test_clip_path)
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=4)

    val_dataloader = DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)