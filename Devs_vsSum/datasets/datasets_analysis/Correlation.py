import os, sys
import glob
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import numpy as np
import datasets.SumMe.path_vars as dataset_pathvars

from FeatureBank.ImageNet import DatasetUtils as ImageNetDatasetUtils
from I3d_Kinetics import DatasetUtils as KDatasetUtils
from FeatureBank.Places import DatasetUtils as PDatasetUtils
from FeatureBank.Moments import DatasetUtils as MDatasetUtils
from datasets.SumMe import LoadLabels
from scipy.stats.stats import pearsonr
FeatureDirecotry = '/home/zwei/datasets/SumMe/features/ImageNet/VGG'
FeatureDirecotry = '/home/zwei/datasets/SumMe/features/Kinetics/I3D'
FeatureDirecotry = '/home/zwei/datasets/SumMe/features/Places/ResNet50'
FeatureDirecotry = '/home/zwei/datasets/SumMe/features/Moments/ResNet50'
get_tags = ImageNetDatasetUtils.get_tags
get_tags = KDatasetUtils.get_tags
get_tags = PDatasetUtils.get_tags
get_tags = MDatasetUtils.get_tags

feature_paths = glob.glob(os.path.join(FeatureDirecotry, '*.npy'))
feature_paths.sort()

# s_feature_path = feature_paths[0]
for video_idx,  s_feature_path in enumerate(feature_paths):
    s_video_name = os.path.basename(s_feature_path)
    s_video_name_stem = s_video_name.split('.')[0]
    s_annotation_path = os.path.join(dataset_pathvars.ground_truth_dir, '{:s}.mat'.format(s_video_name_stem))


    video_features = np.load(s_feature_path)
    labels_mat = LoadLabels.getRawData(s_annotation_path)
    labels = LoadLabels.getUserScores(labels_mat, set1=True)
    avg_labels = np.mean(labels, axis=1)

    # cut
    feature_length = video_features.shape[0]
    label_length = avg_labels.shape[0]
    min_length = min(feature_length, label_length)
    video_features = video_features[0:min_length, :]
    avg_labels = avg_labels[0:min_length]

    correlation = []
    for feature_idx in range(video_features.shape[1]):

        s_feature = video_features[:, feature_idx]
        s_corr, s_p = pearsonr(s_feature, avg_labels)
        correlation.append(s_corr)
    correlation = np.asarray(correlation)
    topN = 5
    top_indices = correlation.argsort()[-topN:][::-1]
    print "{:02d}\t{:s}".format(video_idx, s_video_name_stem)
    for tag_idx,  x in enumerate(top_indices):
        tag = get_tags(x)
        print "\t{:d}\t{:s}\t{:.04f}".format(tag_idx+1, tag, correlation[x])

# print "DEBUG"




