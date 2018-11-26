import os
import glob

dataset_name = 'TVSum'

dataset_dir = os.path.join(os.path.expanduser('~'), 'datasets', dataset_name)
src_video_dir = os.path.join(dataset_dir, 'videos')
dst_frame_dir = os.path.join(dataset_dir, 'frames')
ground_truth_files = os.path.join(dataset_dir, 'matlab/ydata_tvsum50_v7.mat')
file_names = []
src_videos = glob.glob(os.path.join(src_video_dir, '*.mp4'))
for s_annotation_file in src_videos:
    s_filename = os.path.basename(s_annotation_file).split('.')[0]
    file_names.append(s_filename.replace(' ', '_'))

if __name__ == '__main__':
    pass