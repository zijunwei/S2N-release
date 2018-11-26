# the best way would be check the .mat files, but this will very well approximate

import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import glob

ambiguous_class = 'Ambiguous'
user_root = os.path.expanduser('~')
datast_name = 'THUMOS14'
dataset_path = os.path.join(user_root, 'datasets',datast_name)


def getIdNames(path_root=None, isAug=False):
    if path_root is None:
        path_root = os.path.join(dataset_path, 'info')

    th14classids = []
    th14classnames = []
    target_file = open(os.path.join(path_root, 'detclasslist.txt'), 'rb')
    for line in target_file:
        elements = line.split()
        th14classids.append(int(elements[0]))
        th14classnames.append(elements[1])

    if isAug:
        th14classids.append(0)
        th14classnames.append(ambiguous_class)
    return th14classids, th14classnames


def createKeys(classids, classnames):
    id2name = {}
    name2id = {}
    for s_classid, s_classname in zip(classids, classnames):
        id2name[str(s_classid)] = s_classname
        name2id[s_classname] = s_classid

    return id2name, name2id

def IoU(inteval1, inteval2):
    i1 = [min(inteval1), max(inteval1)]
    i2 = [min(inteval2), max(inteval2)]

    b_union = [min(i1[0], i2[0]), max(i1[1], i2[1])]
    b_inter = [max(i1[0], i2[0]), min(i1[1], i2[1])]

    union = b_union[1] - b_union[0]
    intersection = b_inter[1] - b_inter[0]
    return intersection*1./union

if __name__ == '__main__':

    classids, classnames = getIdNames(isAug=True)

    annotation_directory = os.path.join(dataset_path, 'annotations')
    split = 'test'


    video_file_names = []

    for s_classname, class_id in zip(classnames,classids):
        target_filename = '{:s}_{:s}.txt'.format(s_classname, split)
        s_filename = os.path.join(annotation_directory, target_filename)
        s_file = open(s_filename, 'rb')
        for line in s_file:
            elements = line.strip().split()
            video_name = elements[0]
            action_idx = class_id
            start_s = float(elements[1])
            end_s = float(elements[2])
            if video_name in video_file_names:
                continue
            video_file_names.append(video_name)


    frame_rate = 30.0
    frameflow_directory = '/home/zwei/datasets/THUMOS14/frameflow'
    video_lengths = []
    for s_video_name in video_file_names:
        target_directory = os.path.join(frameflow_directory, s_video_name)
        n_files = len(glob.glob(os.path.join(target_directory, '*.jpg')))
        video_length = n_files*1. / 3 /frame_rate
        video_lengths.append(video_length)




    max_length = max(video_lengths)
    min_length = min(video_lengths)
    bin_width = 20
    plt.hist(np.asarray(video_lengths), bins=range(0, int(max_length+bin_width), bin_width))
    plt.title("{:s}#videos: {:d}; MaxLen: {:.2f}, MinLen:{:.2f}".format(split, len(video_lengths), max_length, min_length))
    # plt.tight_layout()
    plt.xlabel('seconds')
    plt.ylabel('# of actions')
    plt.savefig('/home/zwei/datasets/THUMOS14/visualizations/video_len_dist/{:s}.pdf'.format(split))
    plt.close()









