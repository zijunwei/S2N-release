import os
import sys
import numpy as np
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import h5py
import pickle as pkl


def loadKyDataset(dataset_name, file_path=None):

    eval_datasets = ['summe', 'tvsum']
    if dataset_name.lower() not in eval_datasets:
        print "dataset {:s} not defined".format(dataset_name)
        sys.exit(-1)

    # eval_dataset = eval_datasets[1]
    if file_path is None:
        KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
        h5f_path = os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(dataset_name.lower()))
    else:
        h5f_path = file_path
    dataset = h5py.File(h5f_path, 'r')
    # dataset_keys = dataset.keys()
    return dataset

def loadConversion(dataset_name, file_path=None):

    eval_datasets = ['summe', 'tvsum']
    if dataset_name.lower() not in eval_datasets:
        print "dataset {:s} not defined".format(dataset_name)
        sys.exit(-1)

    if file_path is None:
        KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
        conversion_path = os.path.join(KY_dataset_path, '{:s}_name_conversion.pkl'.format(dataset_name.lower()))
    else:
        conversion_path = file_path

    raw2Ky, Ky2raw = pkl.load(open(conversion_path, 'rb'))

    return raw2Ky, Ky2raw


def getKyVideoFrames(kydataset):
    dataset_keys = kydataset.keys()
    video_frames = []
    for key in dataset_keys:
        n_frames = kydataset[key]['n_frames'][()]
        video_frames.append(n_frames)
    return video_frames

def getKyDatasetKeys(dataset):
    return dataset.keys()


def searchKeyByFrame(n_frames, video_frames, dataset_keys):
    target_idx = video_frames.index(n_frames)
    return dataset_keys[target_idx]


def createPositions(nFrames, framerate):
    positions = range(0, nFrames, framerate)
    positions = np.asarray(positions)

    return positions


if __name__ == '__main__':
    dataset = loadKyDataset('TVSum')
    print "DEB"