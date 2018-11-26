# change video names to KY datasets.
import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import h5py

import datasets.TVSum.path_vars as TVSum_dataset_pathvars
from DatasetAnalysis import TVSumMultiViewFeatureLoader as TVSumLoader
import datasets.SumMe.path_vars as SumMe_dataset_pathvars
from DatasetAnalysis import SumMeMultiViewFeatureLoader as SumMeLoader

import pickle as pkl


eval_datasets = ['summe', 'tvsum']
eval_dataset = eval_datasets[0]

pathvars = {'summe': SumMe_dataset_pathvars, 'tvsum': TVSum_dataset_pathvars}
data_loaders = {'summe': SumMeLoader, 'tvsum': TVSumLoader}

s_pathvar = pathvars[eval_dataset]
s_dataloader = data_loaders[eval_dataset]

videofile_stems = s_pathvar.file_names
videofile_stems.sort()



KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
h5f_path = os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(eval_dataset))
save_file_path = os.path.join(KY_dataset_path, '{:s}_name_conversion.pkl'.format(eval_dataset))
dataset = h5py.File(h5f_path, 'r')
dataset_keys = dataset.keys()
n_videos = len(dataset_keys)


def getKyVideoFrames(kydataset):
    dataset_keys = kydataset.keys()
    video_frames = []
    for key in dataset_keys:
        n_frames = dataset[key]['n_frames'][()]
        video_frames.append(n_frames)
    return video_frames


raw2Ky = {}
Ky2raw = {}
video_frames = getKyVideoFrames(dataset)
for video_idx, s_filename in enumerate(videofile_stems):

    video_features, _, _ = s_dataloader.load_by_name(s_filename, doSoftmax=False)
    # sklearn.preprocessing.normalize(video_features)

    n_frames = video_features.shape[0]
    target_idx = video_frames.index(n_frames)
    key = dataset_keys[target_idx]
    # assert n_frames == video_features.shape[0], "Double check"
    raw2Ky[s_filename]=key
    Ky2raw[key]=s_filename

assert len(raw2Ky)==len(Ky2raw)==n_videos, 'Check correspondences'
conversions = [raw2Ky, Ky2raw]
pkl.dump(conversions, open(save_file_path, 'wb'))








