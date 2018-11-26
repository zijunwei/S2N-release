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


def scatter(x, scores, sample_rate=1):
    n_colors = 11
    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", n_colors))
    palette = np.array(sns.cubehelix_palette(n_colors))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # ax.plot(x[:, 0], x[:, 1])

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[(scores*(n_colors-1)).astype(np.int)])

    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    # txts = []
    # for i in range(len(scores)):
    #     if i == 0:
    #         xtext, ytext = x[i, :]
    #         txt = ax.text(xtext, ytext, 'initF', fontsize=12, color='r')
    #         txts.append(txt)
    #         continue
    #
    #     if i % 10 == 0 and i != 0 and i %200 != 0:
    #     # Position of each label.
    #         xtext, ytext = x[i, :]
    #         txt = ax.text(xtext, ytext, '{:.02f}'.format(scores[i]), fontsize=10)
    #         # txt.set_path_effects([
    #         #     PathEffects.Stroke(linewidth=5, foreground="w"),
    #         #     PathEffects.Normal()])
    #         txts.append(txt)
    #         continue
    #
    #     if i % 200 ==0:
    #         xtext, ytext = x[i, :]
    #         txt = ax.text(xtext, ytext, '{:d}'.format(i*sample_rate), fontsize=12, color='r')
    #
    #
    #     if i == len(scores)-1:
    #         xtext, ytext = x[i, :]
    #         txt = ax.text(xtext, ytext, 'endF', fontsize=12, color='r')
    #         txts.append(txt)

    return f, ax, sc


sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()
totalF1 = 0
save_dir = dir_utils.get_dir('t-SNEVisualization')
full_video_features = []
full_video_labels = []

for video_idx, s_filename in enumerate(video_file_stem_names):

    # s_filename = video_file_stem_names[0]
    s_segments = pdefined_segs[s_filename].tolist()
    video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    avg_labels = np.mean(user_labels, axis=1)
    video_features = video_features[::sample_rate, :]
    avg_labels = avg_labels[::sample_rate]
    # print "DB"
    full_video_features.append(video_features)
    full_video_labels.append(avg_labels)

full_video_features = np.vstack(full_video_features)
full_video_labels = np.hstack(full_video_labels)


digits_proj = TSNE(random_state=0).fit_transform(full_video_features)
scatter(digits_proj, full_video_labels, sample_rate)
# ax.title(s_filename)
#
plt.title("Score Distribution")
save_name = os.path.join(save_dir, '{:s}-NonNorm.PNG'.format('Scores'))
plt.savefig(save_name)

    # print "DEBUG"

# print "overall F1 score: {:.04f}".format(totalF1/len(video_file_stem_names))




# print "DEBUG"




