# check the number of extracted frames vs the frames reported in the annotaiton file

import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/VideoSum')
sys.path.append(project_root)
import path_vars
import LoadLabels
import glob

frame_dir = os.path.join(path_vars.dataset_dir, 'frames')
tvsum_gt = LoadLabels.load_annotations()

image_format = 'jpg'
for s_video_name in tvsum_gt:
    s_frame_dir = os.path.join(frame_dir, s_video_name)
    nframes = len(glob.glob(os.path.join(s_frame_dir,'*.{:s}'.format(image_format))))
    video_info = tvsum_gt[s_video_name]
    recorded_nframes = video_info['video_nframes']
    print "{:s}\t{:d}\t{:d}\tdiff: {:d}".format(s_video_name, nframes, recorded_nframes, nframes-recorded_nframes)
