import os
import glob

dataset_name = 'SumMe'

dataset_dir = os.path.join(os.path.expanduser('~'), 'datasets', dataset_name)
src_video_dir = os.path.join(dataset_dir, 'videos')
dst_frame_dir = os.path.join(dataset_dir, 'frames')
ground_truth_dir = os.path.join(dataset_dir, 'GT')
annotation_files = glob.glob(os.path.join(ground_truth_dir, '*.mat'))

file_names = []
for s_annotation_file in annotation_files:
    s_filename = os.path.basename(s_annotation_file).split('.')[0]
    file_names.append(s_filename)


