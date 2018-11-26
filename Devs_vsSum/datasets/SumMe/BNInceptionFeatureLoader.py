import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import numpy as np
import LoadLabels
import path_vars
user_root = os.path.expanduser('~')


FeatureDirecotry = os.path.join(user_root, 'datasets/{:s}/features/ImageNet/BNInception'.format(path_vars.dataset_name))


def unify_ftlbl(feature, label):
    feature_length = feature.shape[0]
    label_length = label.shape[0]
    min_length = min(feature_length, label_length)
    video_features = feature[0:min_length, :]
    labels = label[0:min_length, :]
    return video_features, labels


def load_by_name(video_name):
    # video_feature: n by d where n is the number of frames and d is the feature dimension


    video_features = np.load(os.path.join(FeatureDirecotry,'{:s}.npy'.format(video_name)))

    s_annotation_path = os.path.join(path_vars.ground_truth_dir, '{:s}.mat'.format(video_name))
    labels_mat = LoadLabels.getRawData(s_annotation_path)
    user_labels = LoadLabels.getUserScores(labels_mat, set1=True)

    video_features, user_labels = unify_ftlbl(video_features, user_labels)
    feature_sizes = video_features.shape[1]

    return video_features, user_labels, feature_sizes






if __name__ == '__main__':
    video_name = 'Kids_playing_in_leaves'
    video_features, user_labels, feature_sizes = load_by_name(video_name)
    print "DB"

