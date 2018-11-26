import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import matplotlib as mpl

mpl.use('Agg')
import numpy as np
import pandas as pd
import pickle
import progressbar
from ActionLocalizationDevs.PropEval.obselete import prop_eval_utils_mod as prop_eval_utils


#
# prediction_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/pkl_files/TURN-C3D-16_thumos14.pkl'
# ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'



def pkl_seconds2dataframe(frm_nums):
    data_frame = []
    # movie_fps = pickle.load(open("./movie_fps.pkl"))
    # pkl_dir = "./pkl_files/"
    dt_results = pickle.load(open(prediction_file))
    pbar = progressbar.ProgressBar(max_value=len(dt_results))
    for i, _key in enumerate(dt_results):
        pbar.update(i)
        # fps = movie_fps[_key]
        frm_num = frm_nums[_key]
        for line in dt_results[_key]:
            start = int(line[0] * 30)
            end = int(line[1] * 30)
            score = float(line[2])
            data_frame.append([end, start, score, frm_num, _key])
    return data_frame

def pkl_frame2dataframe(frm_nums):
    data_frame = []
    # movie_fps = pickle.load(open("./movie_fps.pkl"))
    # pkl_dir = "./pkl_files/"
    dt_results = pickle.load(open(prediction_file))
    pbar = progressbar.ProgressBar(max_value=len(dt_results))
    for i, _key in enumerate(dt_results):
        pbar.update(i)
        # fps = movie_fps[_key]
        frm_num = frm_nums[_key]
        for line in dt_results[_key]:
            start = int(line[0])
            end = int(line[1])
            score = float(line[2])
            data_frame.append([end, start, score, frm_num, _key])
    return data_frame


# save_name = 'lstm2heads_0291'
save_name = 'TURN-FLOW-16'
freq=0.2
framebased = False

prediction_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/pkl_files/{:s}_thumos14.pkl'.format(save_name)
ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'
frm_nums = pickle.load(open("./frm_num.pkl"))
if framebased:
    rows = pkl_frame2dataframe(frm_nums)

else:
    rows = pkl_seconds2dataframe(frm_nums)

daps_results = pd.DataFrame(rows, columns=['f-end', 'f-init', 'score', 'video-frames', 'video-name'])

# Retrieves and loads Thumos14 test set ground-truth.
# ground_truth_url = ('https://gist.githubusercontent.com/cabaf/'
#                     'ed34a35ee4443b435c36de42c4547bd7/raw/'
#                     '952f17b9cdc6aa4e6d696315ba75091224f5de97/'
#                     'thumos14_test_groundtruth.csv')
# s = requests.get(ground_truth_url).content
ground_truth = pd.read_csv(ground_truth_file, sep=' ')
# Computes average recall vs average number of proposals.


recall_freq, tiou_thresholds_freq = prop_eval_utils.recall_freq_vs_tiou_thresholds(daps_results, ground_truth, frm_nums,
                                                                                   pdefined_freq=freq)




recall_freq_pnt_file = "./ref_pnt_pairs/{:s}_{:.2f}_recall_freq_pnt_pairs.npy"
np.save(recall_freq_pnt_file.format(save_name, freq), np.array([tiou_thresholds_freq, recall_freq]))


