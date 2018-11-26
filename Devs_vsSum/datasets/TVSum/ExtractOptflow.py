import os
import sys

project_root = os.path.join(os.path.expanduser('~'), 'Dev/VideoSum')
sys.path.append(project_root)

import PyUtils.dir_utils as dir_utils
import glob
import subprocess
import path_vars

dataset_dir = path_vars.dataset_dir
src_video_dir = os.path.join(dataset_dir, 'videos')
dst_frame_dir = dir_utils.get_dir(os.path.join(dataset_dir, 'optflow'))

src_videopath_list = glob.glob(os.path.join(src_video_dir, '*.mp4'))

for i, s_src_videopath in enumerate(src_videopath_list):

    if not os.path.isfile(s_src_videopath):
        print "Not Exist: {:s}".format(s_src_videopath)
        continue
    video_stem = os.path.splitext(os.path.basename(s_src_videopath))[0]
    print "[{:02d} | {:02d}], {:s}".format(i, len(src_videopath_list), video_stem)
    dst_framepath = dir_utils.get_dir(os.path.join(dst_frame_dir, video_stem))
    exe = './denseFlow_gpu'
    command = [exe,
               '-f="%s"' % s_src_videopath,
               '-b=20', '-t=1', '-d=0', '-s=1', '-h=0', '-w=0',
               '-x="%s/x"' % (dst_framepath),
               '-y="%s/y"' % (dst_framepath),
               '-i="%s/i"' % (dst_framepath)
               ]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print err.output








