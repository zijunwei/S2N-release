import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import numpy as np
import vsSummDevs.datasets.SumMe.path_vars as dataset_pathvars
from vsSummDevs.datasets.SumMe import LoadLabels
user_root = os.path.expanduser('~')
import math
from PyUtils.softmax import softmax

FeatureDirecotry = [os.path.join(user_root, 'datasets/SumMe/features/ImageNet/VGG')]
FeatureDirecotry.append(os.path.join(user_root, 'datasets/SumMe/features/Kinetics/I3D'))
FeatureDirecotry.append(os.path.join(user_root, 'datasets/SumMe/features/Places/ResNet50'))
FeatureDirecotry.append(os.path.join(user_root, 'datasets/SumMe/features/Moments/ResNet50'))



def unify_ftlbl(feature, label):
    feature_length = feature.shape[0]
    label_length = label.shape[0]
    min_length = min(feature_length, label_length)
    video_features = feature[0:min_length, :]
    labels = label[0:min_length, :]
    return video_features, labels




def feature_entropy(video_features, feature_sizes=None):
    if feature_sizes is None:
        feature_sizes = [video_features.shape[1]]
    total_feature_dims = sum(feature_sizes)
    acc_feature_size = [0]
    acc = 0
    for s_feature_size in feature_sizes:
        acc += s_feature_size
        acc_feature_size.append(acc)
    assert acc_feature_size[-1] == total_feature_dims, 'check feature entropy check'
    feature_segments = []
    for i in range(len(acc_feature_size)-1):
        cur_seg = [acc_feature_size[i], acc_feature_size[i+1]]
        feature_segments.append(cur_seg)

    acc_feature_entropy = np.zeros(video_features.shape[0])
    for s_seg in feature_segments:
        selected_feature = video_features[:, s_seg[0]:s_seg[1]]
        acc_feature_entropy += -np.sum(selected_feature * np.log(selected_feature), axis=1)
    return acc_feature_entropy


def load_by_name(video_name, doSoftmax=False, featureset=None):
    # video_feature: n by d where n is the number of frames and d is the feature dimension
    video_features = []
    feature_sizes = []
    if featureset is None:
        selectedFeatureDirecotry = FeatureDirecotry
    else:
        selectedFeatureDirecotry = FeatureDirecotry[featureset[0]:featureset[1]]
    for s_dir in selectedFeatureDirecotry:
        s_video_features = np.load(os.path.join(s_dir,'{:s}.npy'.format(video_name)))
        if doSoftmax:
            s_video_features = softmax(s_video_features, axis=1)
        feature_sizes.append(s_video_features.shape[1])
        video_features.append(s_video_features)
    video_features = np.hstack(video_features)
    s_annotation_path = os.path.join(dataset_pathvars.ground_truth_dir, '{:s}.mat'.format(video_name))
    labels_mat = LoadLabels.getRawData(s_annotation_path)
    user_labels = LoadLabels.getUserScores(labels_mat, set1=True)

    video_features, user_labels = unify_ftlbl(video_features, user_labels)
    return video_features, user_labels, feature_sizes


def PositionEncoddings(nfeatures, feature_dimensions, base=10000):
    position_features = np.zeros([nfeatures, feature_dimensions])
    for i in range(feature_dimensions):
        for pos in range(nfeatures):
            if i % 2 == 0:
                position_features[pos, i] = math.sin(pos*1./pow(base,i*1./feature_dimensions))
            else:
                position_features[pos, i] = math.cos(pos*1./pow(base,i*1./feature_dimensions))
    return position_features



if __name__ == '__main__':
    video_name = 'Kids_playing_in_leaves'
    video_features, user_labels, feature_sizes = load_by_name(video_name, doSoftmax=True)
    print "DB"

