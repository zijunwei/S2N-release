import glob
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import progressbar

import LoadLabels
import PyUtils.dir_utils as dir_utils
import path_vars
from datasets.SingleVideoFrameDataset import default_loader


def vis_summaries(image_list, user_scores, save_target=None, everyN=30):
    n_frames, n_users = user_scores.shape
    min_frames = min(len(image_list), n_frames)
    image_list = image_list[:n_frames]
    user_scores = user_scores[:n_frames,:]
    average_user_scores = np.mean(user_scores, axis=1)
    n_plots = 2
    pbar = progressbar.ProgressBar(max_value=min_frames)
    image_idx = 0
    for i, s_image_path in enumerate(image_list):

        pbar.update(i)
        if i % everyN != 0:
            continue

        s_image = default_loader(s_image_path)
        ax = plt.subplot(n_plots,1,1)
        plt.imshow(np.array(s_image))
        ax.axis('off')
        plt.title('F:{:08d}  S:{:.04f}'.format(i, average_user_scores[i]))
        ax2 = plt.subplot(n_plots,1,2)
        plt.plot(range(min_frames), average_user_scores, '-')
        ax2.set_ylim([0, 5])
        plt.annotate('', xy=(i, average_user_scores[i]), xytext=(i, average_user_scores[i]+0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )
        if save_target is None:
            plt.show()
        else:
            save_target = dir_utils.get_dir(save_target)
            plt.savefig(os.path.join(save_target, '{:08d}.jpg'.format(image_idx)))

        plt.close()
        image_idx += 1


if __name__ == '__main__':
    frame_dir = os.path.join(path_vars.dataset_dir, 'frames')
    tvsum_gt = LoadLabels.load_annotations()
    save_directory = dir_utils.get_dir(os.path.join(path_vars.dataset_dir, 'visualizations'))
    image_foramt = 'jpg'

    for video_idx, s_video_name in enumerate(tvsum_gt):
        # s_name = os.path.splitext(os.path.basename(s_annotation_file))[0]
        print "[{:d} | {:d}]".format(video_idx, len(tvsum_gt))
        s_image_directory = os.path.join(frame_dir, s_video_name)
        image_pathlist = glob.glob(os.path.join(s_image_directory, '*.{:s}'.format(image_foramt)))
        image_pathlist.sort()
        s_video_info = tvsum_gt[s_video_name]
        s_FPS = s_video_info['video_nframes']/s_video_info['video_seg_seconds']
        user_scores = s_video_info['video_user_scores']
        s_save_directory = os.path.join(save_directory, s_video_name)
        vis_summaries(image_pathlist, user_scores, save_target=s_save_directory)

        ffmpeg_command = ["ffmpeg",  "-r", str(s_FPS), "-i", '"{:s}"'.format(os.path.join(s_save_directory, "%08d.{:s}".format(image_foramt))),
                          "-vcodec", "mpeg4", "-q:v", "5", '"{:s}"'.format(os.path.join(save_directory, "{:s}.mp4".format(s_video_name)))]
        ffmpeg_command = ' '.join(ffmpeg_command)
        subprocess.call(ffmpeg_command, shell=True)

