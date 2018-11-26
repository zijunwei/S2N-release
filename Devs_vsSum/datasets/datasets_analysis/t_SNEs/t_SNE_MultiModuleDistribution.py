# Selecting based on entropy...
# compute the entropy of each elements, each frame is given an entropy score
import os
import sys

import numpy as np

import datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import datasets.getShotBoundariesECCV2016 as getSegs
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from PyUtils import dir_utils


def subscatter(ax, x, scores, sample_rate=1):
    n_colors = 11
    palette = np.array(sns.cubehelix_palette(n_colors))



    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[(scores*(n_colors-1)).astype(np.int)])

    ax.axis('equal')
    return


sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()
totalF1 = 0
save_dir = dir_utils.get_dir('t-SNEVisualization')


for video_idx, s_filename in enumerate(video_file_stem_names):
    print '[{:d} | {:d}], {:s}'.format(video_idx, len(video_file_stem_names), s_filename)
    video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=True)
    avg_labels = np.mean(user_labels, axis=1)
    video_features = video_features[::sample_rate, :]
    avg_labels = avg_labels[::sample_rate]

    feature_types = ['ImageNet','Kinectics','Places', 'Moments']
    f = plt.figure(figsize=(8, 8))
    feature_sizes = np.cumsum(np.asarray(feature_sizes))
    # for curIdx in range(video_features.shape[0]):
    for i in range(len(feature_sizes)):
        if i == 0:
            segs = [0, feature_sizes[i]]
        else:
            segs = [feature_sizes[i-1], feature_sizes[i]]
        ax = plt.subplot(2, 2, i+1)
        sub_features = video_features[:, segs[0]:segs[1]]
        digits_proj = TSNE(random_state=0).fit_transform(sub_features)
        subscatter(ax, digits_proj, avg_labels, sample_rate)
        plt.title(feature_types[i])
# ax.title(s_filename)
#
# plt.title("Score Distribution")
    save_name = os.path.join(save_dir, '{:s}-4SepNorm.PNG'.format(s_filename))
    plt.savefig(save_name)

    # print "DEBUG"

# print "overall F1 score: {:.04f}".format(totalF1/len(video_file_stem_names))




# print "DEBUG"




