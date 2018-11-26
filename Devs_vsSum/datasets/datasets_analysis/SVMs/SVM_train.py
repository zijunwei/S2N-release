import glob
import os
import sys

import numpy as np

import datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
from datasets.SumMe import LoadLabels
from scipy.stats.stats import pearsonr
from sklearn import svm
import datasets.getShotBoundariesECCV2016 as getSegs
FeatureDirecotry = ['/home/zwei/datasets/SumMe/features/ImageNet/VGG']
FeatureDirecotry.append('/home/zwei/datasets/SumMe/features/Kinetics/I3D')
FeatureDirecotry.append('/home/zwei/datasets/SumMe/features/Places/ResNet50')
FeatureDirecotry.append('/home/zwei/datasets/SumMe/features/Moments/ResNet50')
from vsSummDevs.SumEvaluation import rep_conversions, metrics

# get_tags = ImageNetDatasetUtils.get_tags
# get_tags = KDatasetUtils.get_tags
# get_tags = PDatasetUtils.get_tags
# get_tags = MDatasetUtils.get_tags

feature_paths = glob.glob(os.path.join(FeatureDirecotry[0], '*.npy'))
feature_paths.sort()

def get_feature_matrix(feature_file):
    video_features = np.load(s_feature_path)
    return video_features


def unify_ftavglbl(feature, label):
    # cut
    feature_length = feature.shape[0]
    label_length = label.shape[0]
    min_length = min(feature_length, label_length)
    video_features = feature[0:min_length, :]
    avg_labels = label[0:min_length]
    return video_features, avg_labels

def unify_ftlbl(feature, label):
    feature_length = feature.shape[0]
    label_length = label.shape[0]
    min_length = min(feature_length, label_length)
    video_features = feature[0:min_length, :]
    labels = label[0:min_length, :]
    return video_features, labels

# s_feature_path = feature_paths[0]

pdefined_segs = getSegs.getSumMeShotBoundaris()

# train 0-15,
# val 16:20
# test: 21-25

train_video_features = []
train_video_labels = []
for video_idx in range(10):
    s_feature_path = feature_paths[video_idx]
    s_video_name = os.path.basename(s_feature_path)
    s_video_name_stem = s_video_name.split('.')[0]
    s_annotation_path = os.path.join(dataset_pathvars.ground_truth_dir, '{:s}.mat'.format(s_video_name_stem))

    video_features = []
    for s_dir in FeatureDirecotry:
        s_video_features = np.load(os.path.join(s_dir,'{:s}.npy'.format(s_video_name_stem)))
        video_features.append(s_video_features)

    video_features = np.hstack(video_features)
    labels_mat = LoadLabels.getRawData(s_annotation_path)
    user_labels = LoadLabels.getUserScores(labels_mat, set1=True)
    avg_labels = np.mean(user_labels, axis=1)

    # cut
    feature_length = video_features.shape[0]
    label_length = user_labels.shape[0]
    min_length = min(feature_length, label_length)
    video_features = video_features[0:min_length, :]
    avg_labels = avg_labels[0:min_length]
    user_labels = user_labels[0:min_length, :]

    train_video_labels.append(avg_labels)
    train_video_features.append(video_features)

train_video_features = np.vstack(train_video_features)
train_video_labels = np.hstack(train_video_labels)


clf = svm.LinearSVR()
clf.fit(train_video_features, train_video_labels)

F1_scores = 0

for video_idx in range(15, 25):
    s_feature_path = feature_paths[video_idx]
    s_video_name = os.path.basename(s_feature_path)
    s_video_name_stem = s_video_name.split('.')[0]
    s_annotation_path = os.path.join(dataset_pathvars.ground_truth_dir, '{:s}.mat'.format(s_video_name_stem))

    video_features = []
    for s_dir in FeatureDirecotry:
        s_video_features = np.load(os.path.join(s_dir, '{:s}.npy'.format(s_video_name_stem)))
        video_features.append(s_video_features)

    video_features = np.hstack(video_features)
    labels_mat = LoadLabels.getRawData(s_annotation_path)
    user_labels = LoadLabels.getUserScores(labels_mat, set1=True)
    avg_labels = np.mean(user_labels, axis=1)

    # cut
    feature_length = video_features.shape[0]
    label_length = avg_labels.shape[0]
    min_length = min(feature_length, label_length)
    video_features = video_features[0:min_length, :]
    avg_labels = avg_labels[0:min_length]
    user_labels = user_labels[0:min_length, :]


    frame_contrib = clf.predict(video_features)
    frame_contrib = (frame_contrib- np.min(frame_contrib))/(np.max(frame_contrib)-np.mean(frame_contrib))
    s_seg = pdefined_segs[s_video_name_stem]

    s_frame01scores = rep_conversions.framescore2frame01score(frame_contrib, s_seg)
    user_scores_list = [user_labels[:, i] for i in range(user_labels.shape[1])]
    s_F1_score = metrics.averaged_F1_score(y_trues=user_scores_list, y_score=s_frame01scores.tolist())

    s_scorr, s_p = pearsonr(frame_contrib, avg_labels)
    print  "[{:d} | {:d}] \t {:s} \t{:.04f}\t correlation:{:.04f}".format(video_idx, len(feature_paths), s_video_name_stem, s_F1_score, s_scorr)
    F1_scores += s_F1_score


print "overall F1 score: {:.04f}".format(F1_scores/10)




# print "DEBUG"




