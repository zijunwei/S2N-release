import os
import pandas as pd
import numpy as np
import pickle as pkl
import path_vars

def ReadAnnotations(annotation_file=None, unit_size=16., class_list=None):
    if annotation_file is None:
        annotation_file = '/home/zwei/Dev/TURN_TAP_ICCV17/turn_codes/val_training_samples.txt'
    movie_instances = {}

    with open(annotation_file) as f:
        for l in f:

            movie_name = l.rstrip().split(" ")[0]
            gt_start = float(l.rstrip().split(" ")[3])
            gt_end = float(l.rstrip().split(" ")[4])

            action_category = l.rstrip().split(" ")[-1]
            class_id = class_list.index(action_category)
            if movie_name in movie_instances.keys():
                movie_instances[movie_name].append((gt_start, gt_end, class_id))
            else:
                movie_instances[movie_name] = [(gt_start, gt_end, class_id)]

    return movie_instances


save_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_val_groundtruth.csv'


data_frame = []

pathVars = path_vars.PathVars()

movie_instances = ReadAnnotations(class_list=pathVars.classnames)

for i, _key in enumerate(movie_instances):

    # fps = movie_fps[_key]

    frm_num = pathVars.video_frames[_key]
    for line in movie_instances[_key]:
        start = int(line[0])
        end = int(line[1])
        label_idx = int(line[2])
        data_frame.append([end, start, label_idx, frm_num, _key])

results = pd.DataFrame(data_frame, columns=['f-end', 'f-init', 'label-idx', 'video-frames', 'video-name'])
results.to_csv(save_file, sep=' ', index=False)
