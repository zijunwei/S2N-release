import sys, os
import shutil
import glob
import progressbar


split = 'test'
src_directory = '/home/zwei/datasets/THUMOS14/features/DenseFlowICCV2017/{:s}_fc6_16_overlap0.5_denseflow'.format(split)
target_directory = '/home/zwei/datasets/THUMOS14/features/denseflow'

src_files = glob.glob(os.path.join(src_directory, '*.npy'))
pbar = progressbar.ProgressBar(max_value=len(src_files))
for idx, s_file in enumerate(src_files):
    pbar.update(idx)
    s_filename = os.path.basename(s_file)
    target_file = os.path.join(target_directory, s_filename)
    shutil.copyfile(s_file, target_file)