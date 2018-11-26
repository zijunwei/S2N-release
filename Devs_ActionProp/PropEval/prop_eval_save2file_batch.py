import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import io
import requests
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import progressbar
import prop_eval_utils
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

for file_idx in range(1, 300, 10):
    save_name = 'lstm2heads_{:04d}'.format(file_idx)
    framebased = True

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
    average_recall, average_nr_proposals = prop_eval_utils.average_recall_vs_nr_proposals(daps_results,
                                                                          ground_truth)

    # Computes average recall vs proposal frequency.
    average_recall_freq, freqs = prop_eval_utils.average_recall_vs_freq(daps_results, ground_truth, frm_nums)

    # Computes recall for different tiou thresholds at a fixed average number of proposals.
    recall, tiou_thresholds = prop_eval_utils.recall_vs_tiou_thresholds(daps_results, ground_truth,
                                                        nr_proposals=1000)

    recall_freq, tiou_thresholds_freq = prop_eval_utils.recall_freq_vs_tiou_thresholds(daps_results, ground_truth, frm_nums)



    avg_prop_pnt_file = "./ref_pnt_pairs/{:s}_avg_prop_pnt_pairs.npy"
    np.save(avg_prop_pnt_file.format(save_name), np.array([average_nr_proposals, average_recall]))

    freq_pnt_file = "./ref_pnt_pairs/{:s}_freq_pnt_pairs.npy"
    np.save(freq_pnt_file.format(save_name), np.array([freqs, average_recall_freq]))

    recall1000_pnt_file = "./ref_pnt_pairs/{:s}_recall_pnt_pairs.npy"

    np.save(recall1000_pnt_file.format(save_name), np.array([tiou_thresholds, recall]))

    recall_freq_pnt_file = "./ref_pnt_pairs/{:s}_recall_freq_pnt_pairs.npy"
    np.save(recall_freq_pnt_file.format(save_name), np.array([tiou_thresholds_freq, recall_freq]))


