import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/VideoSum')
sys.path.append(project_root)

import path_vars
import scipy.io as s_io
import numpy as np
dataset_dir = path_vars.dataset_dir


def load_annotations():
    annotation_file = os.path.join(dataset_dir, 'matlab', 'ydata_tvsum50_v7.mat')
    annotation_mat = s_io.loadmat(annotation_file)
    tvsum = annotation_mat['tvsum50'][0]
    tvsum_gt = {}
    for s_entry in tvsum:
        video_info = {}
        video_info['video_name'] = s_entry[0][0].replace(' ','_')
        video_info['video_category'] = s_entry[1][0]
        video_info['video_title'] = s_entry[2][0]
        video_info['video_seg_seconds'] = s_entry[3][0][0]
        video_info['video_nframes'] = s_entry[4][0][0]
        video_info['video_user_scores'] = s_entry[5]
        video_info['video_avg_score'] = s_entry[6]
        tvsum_gt[video_info['video_name']] = video_info

    return tvsum_gt

if __name__ == '__main__':
    tvsum_gt = load_annotations()
    for s_video_name in tvsum_gt:
        s_video_info = tvsum_gt[s_video_name]
        s_FPS = s_video_info['video_nframes']/s_video_info['video_seg_seconds']
        print "{:s}\tFPS:{:.04f}".format(s_video_name, s_FPS)
    print "DEBUG"


