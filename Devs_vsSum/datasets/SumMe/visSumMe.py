import glob
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.io as s_io

import LoadLabels
import PyUtils.dir_utils as dir_utils
from datasets.SingleVideoFrameDataset import default_loader


def vis_summaries(image_list, user_scores, save_target=None):
    n_frames, n_users = user_scores.shape
    min_frames = min(len(image_list), n_frames)
    image_list = image_list[:n_frames]
    user_scores = user_scores[:n_frames,:]
    average_user_scores = np.mean(user_scores, axis=1)
    n_plots = 2
    pbar = progressbar.ProgressBar(max_value=min_frames)
    for i, s_image_path in enumerate(image_list):
        pbar.update(i)
        s_image = default_loader(s_image_path)
        ax = plt.subplot(n_plots,1,1)
        plt.imshow(np.array(s_image))
        ax.axis('off')
        plt.title('F:{:08d}  S:{:.04f}'.format(i, average_user_scores[i]))
        plt.subplot(n_plots,1,2)
        plt.plot(range(min_frames), average_user_scores, '-')
        plt.annotate('', xy=(i, average_user_scores[i]), xytext=(i, average_user_scores[i]+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )
        if save_target is None:
            plt.show()
        else:
            save_target = dir_utils.get_dir(save_target)
            plt.savefig(os.path.join(save_target, os.path.basename(s_image_path)))

        plt.close()



if __name__ == '__main__':

    image_directory = '/home/zwei/datasets/SumMe/frames'
    annotation_directory = '/home/zwei/datasets/SumMe/GT'
    save_directory = '/home/zwei/datasets/SumMe/Visualization'
    annotation_format = 'mat'
    image_foramt = 'jpg'
    annotation_file_list = glob.glob(os.path.join(annotation_directory, '*.{:s}'.format(annotation_format)))


    for video_idx, s_annotation_file in enumerate(annotation_file_list):
        s_name = os.path.splitext(os.path.basename(s_annotation_file))[0]
        print "[{:d} | {:d}]".format(video_idx, len(annotation_file_list))
        s_image_directory = os.path.join(image_directory, s_name)
        image_pathlist = glob.glob(os.path.join(s_image_directory, '*.{:s}'.format(image_foramt)))
        image_pathlist.sort()
        annotation_mat = s_io.loadmat(s_annotation_file)
        fps = annotation_mat['FPS'][0][0]
        user_scores = LoadLabels.getUserScores(annotation_mat)
        s_save_directory = os.path.join(save_directory, s_name.replace(' ', '_'))
        vis_summaries(image_pathlist, user_scores, save_target=s_save_directory)

        ffmpeg_command = ["ffmpeg",  "-r", str(fps), "-i", '"{:s}"'.format(os.path.join(s_save_directory, "%08d.{:s}".format(image_foramt))),
                          "-vcodec", "mpeg4", "-q:v", "5", '"{:s}"'.format(os.path.join(save_directory, "{:s}.mp4".format(s_name.replace(' ', '_'))))]
        ffmpeg_command = ' '.join(ffmpeg_command)
        subprocess.call(ffmpeg_command, shell=True)

