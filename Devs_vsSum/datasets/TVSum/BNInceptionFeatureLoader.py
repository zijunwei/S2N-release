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

tvsum_gt = LoadLabels.load_annotations()

def load_by_name(video_name):
    video_features = np.load(os.path.join(FeatureDirecotry,'{:s}.npy'.format(video_name)))
    user_labels = tvsum_gt[video_name]['video_user_scores']
    feature_sizes = video_features.shape[1]
    video_features, user_labels = unify_ftlbl(video_features, user_labels)
    return video_features, user_labels, feature_sizes


if __name__ == '__main__':
    video_name = '0tmA_C6XwfM'
    video_features, user_labels, feature_sizes = load_by_name(video_name)
    print "DB"

