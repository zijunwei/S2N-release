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
import subprocess
import progressbar

def subscatter(ax, curIdx, x, scores, sample_rate=1):
    n_colors = 11
    palette = np.array(sns.cubehelix_palette(n_colors))
    plot_lag = 10

    startidx = max(0, curIdx-plot_lag)
    ax.plot(x[startidx:curIdx, 0], x[startidx:curIdx, 1])
    if curIdx>1:
        ax.plot(x[curIdx-1:curIdx, 0], x[curIdx-1:curIdx, 1], color='k')


    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[(scores*(n_colors-1)).astype(np.int)])

    ax.axis('equal')
    ax.set_xlim([-80, 80])
    ax.set_ylim([-80, 80])
    # ax.xlim(-80, 80)
    # ax.ylim(-80, 80)
    return


sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()
totalF1 = 0
save_dir = dir_utils.get_dir('t-SNEVisualization')


for video_idx, s_filename in enumerate(video_file_stem_names):
    print '[{:d} | {:d}], {:s}'.format(video_idx, len(video_file_stem_names), s_filename)
    video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    avg_labels = np.mean(user_labels, axis=1)
    video_features = video_features[::sample_rate, :]
    avg_labels = avg_labels[::sample_rate]

    feature_types = ['ImageNet','Kinectics','Places', 'Moments']
    feature_sizes = np.cumsum(np.asarray(feature_sizes))
    s_save_dir = dir_utils.get_dir(os.path.join(save_dir, s_filename+'-Sep4'))
    pbar= progressbar.ProgressBar(max_value=video_features.shape[0])


    digits_projs = []
    for i in range(len(feature_sizes)):
        if i == 0:
            segs = [0, feature_sizes[i]]
        else:
            segs = [feature_sizes[i-1], feature_sizes[i]]

        sub_features = video_features[:, segs[0]:segs[1]]
        digits_proj = TSNE(random_state=0).fit_transform(sub_features)
        digits_projs.append(digits_proj)

    for curIdx in range(video_features.shape[0]):
        pbar.update(curIdx)
        plt.figure(figsize=(8, 8))
        for j in range(len(feature_sizes)):
            ax = plt.subplot(2, 2, j+1)
            subscatter(ax, curIdx, digits_projs[j], avg_labels, sample_rate)
            plt.title(feature_types[j])

        save_name = os.path.join(s_save_dir, '{:04d}.PNG'.format(curIdx))
        plt.savefig(save_name)
        plt.close()
    ffmpeg_command = ["ffmpeg", "-r", str(25), "-i",
                      '"{:s}"'.format(os.path.join(s_save_dir, "%04d.{:s}".format('PNG'))),
                      "-vcodec", "mpeg4", "-q:v", "5", '"{:s}"'.format(
            os.path.join(save_dir, "{:s}-Sep4.mp4".format(s_filename.replace(' ', '_'))))]
    ffmpeg_command = ' '.join(ffmpeg_command)
    subprocess.call(ffmpeg_command, shell=True)

    # print "DEBUG"

# print "overall F1 score: {:.04f}".format(totalF1/len(video_file_stem_names))




# print "DEBUG"




