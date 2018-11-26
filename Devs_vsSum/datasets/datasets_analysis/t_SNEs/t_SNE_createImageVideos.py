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
from PIL import Image
def default_loader(path):
    return Image.open(path).convert('RGB')
def scatter(x, scores, s_image, currentIdx, sample_rate=1):
    n_colors = 11
    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", n_colors))
    palette = np.array(sns.cubehelix_palette(n_colors))
    plot_lag=10
    # We create a scatter plot.
    # f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 1, 1)
    plt.imshow(s_image)
    ax.axis('off')
    ax = plt.subplot(2,1,2, aspect='equal')

    startidx = max(0, currentIdx-plot_lag)
    ax.plot(x[startidx:currentIdx, 0], x[startidx:currentIdx, 1])
    if currentIdx>1:
        ax.plot(x[currentIdx-1:currentIdx, 0], x[currentIdx-1:currentIdx, 1], color='k')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[(scores*(n_colors-1)).astype(np.int)])

    plt.xlim(-80, 80)
    plt.ylim(-80, 80)
    # ax.axis('off')
    # ax.axis('tight')
    # ax.axis('equal')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))

    # ax.set(adjustable='box-forced', aspect='equal')

    # We add the labels for each digit.
    txts = []
    for i in range(len(scores)):
        if i == 0:
            xtext, ytext = x[i, :]
            txt = ax.text(xtext, ytext, 'initF', fontsize=12, color='r')
            txts.append(txt)
            continue

        # if i % 10 == 0 and i != 0 and i %200 != 0 and i <= currentIdx:
        # # Position of each label.
        #     xtext, ytext = x[i, :]
        #     txt = ax.text(xtext, ytext, '{:.02f}'.format(scores[i]), fontsize=10)
        #     # txt.set_path_effects([
        #     #     PathEffects.Stroke(linewidth=5, foreground="w"),
        #     #     PathEffects.Normal()])
        #     txts.append(txt)
        #     continue

        if i % 200 ==0 and i <= currentIdx:
            xtext, ytext = x[i, :]
            txt = ax.text(xtext, ytext, '{:d}'.format(i*sample_rate), fontsize=12, color='r')


        if i == len(scores)-1:
            xtext, ytext = x[i, :]
            txt = ax.text(xtext, ytext, 'endF', fontsize=12, color='r')
            txts.append(txt)

    # return f, ax, sc, txts


sample_rate = 5
pdefined_segs = getSegs.getSumMeShotBoundaris()
video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()
totalF1 = 0
save_dir = dir_utils.get_dir('t-SNEVisualization')
for video_idx, s_filename in enumerate(video_file_stem_names):
    # if video_idx ==0:
    #     continue
    print '{:d} | {:d} \t {:s}'.format(video_idx, len(video_file_stem_names), s_filename)
    # s_filename = video_file_stem_names[0]
    s_segments = pdefined_segs[s_filename].tolist()
    video_features, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    avg_labels = np.mean(user_labels, axis=1)
    video_features = video_features[::sample_rate, :]
    avg_labels = avg_labels[::sample_rate]
    # print "DB"
    digits_proj = TSNE(random_state=0).fit_transform(video_features)
    s_save_dir = dir_utils.get_dir(os.path.join(save_dir, s_filename+'-NonNorm-Image'))
    image_dir = os.path.join('/home/zwei/datasets/SumMe/frames', s_filename)
    pbar= progressbar.ProgressBar(max_value=video_features.shape[0])
    for curIdx in range(video_features.shape[0]):

        s_image = default_loader(os.path.join(image_dir, '{:08d}.jpg'.format(curIdx*sample_rate+1)))
        pbar.update(curIdx)
        scatter(digits_proj, avg_labels, s_image, curIdx, sample_rate)
        # ax.title(s_filename)
        #
        plt.title(s_filename+'  {:05d}'.format(curIdx*sample_rate+1))
        save_name = os.path.join(s_save_dir, '{:04d}.PNG'.format(curIdx))
        plt.savefig(save_name)
        plt.clf()
        plt.close('all')
    ffmpeg_command = ["ffmpeg",  "-r", str(25), "-i", '"{:s}"'.format(os.path.join(s_save_dir, "%04d.{:s}".format('PNG'))),
                      "-vcodec", "mpeg4", "-q:v", "5", '"{:s}"'.format(os.path.join(save_dir, "{:s}-NonNorm-Image2.mp4".format(s_filename.replace(' ', '_'))))]
    ffmpeg_command = ' '.join(ffmpeg_command)
    subprocess.call(ffmpeg_command, shell=True)

    # print "DEBUG"

# print "overall F1 score: {:.04f}".format(totalF1/len(video_file_stem_names))




# print "DEBUG"




