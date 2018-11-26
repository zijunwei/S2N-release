# same as _ourmethods except that using the ground truth generated by ICCV2017 https://github.com/yjxiong/action-detection
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
import ActionLocalizationDevs.PropEval.prop_eval_utils_mod2 as prop_eval_utils


baseline_name = 'inception-s5_15output-fast-0019'

prediction_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/baselines_results/{:s}_thumos14_test.csv'.format(baseline_name)
ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/action_det_prep/thumos14_tag_test_proposal_list.csv'




daps_results = pd.read_csv(prediction_file, sep=' ')
ground_truth = pd.read_csv(ground_truth_file, sep=' ')

target_video_frms = ground_truth[['video-name', 'video-frames']].drop_duplicates().values
frm_nums = {}
for s_target_videofrms in target_video_frms:
    frm_nums[s_target_videofrms[0]] = s_target_videofrms[1]

average_recall, average_nr_proposals = prop_eval_utils.average_recall_vs_nr_proposals(daps_results,
                                                                      ground_truth)
average_recall_freq, freqs = prop_eval_utils.average_recall_vs_freq(daps_results, ground_truth, frm_nums)

recall, tiou_thresholds = prop_eval_utils.recall_vs_tiou_thresholds(daps_results, ground_truth,
                                                    nr_proposals=1000)

recall_freq, tiou_thresholds_freq = prop_eval_utils.recall_freq_vs_tiou_thresholds(daps_results, ground_truth, frm_nums,
                                                                                   pdefined_freq=1.0)



avg_prop_pnt_file = "./baseline_pnt_pairs/{:s}-newtst_avg_prop_pnt_pairs.npy"
np.save(avg_prop_pnt_file.format(baseline_name), np.array([average_nr_proposals, average_recall]))

freq_pnt_file = "./baseline_pnt_pairs/{:s}-newtst_freq_pnt_pairs.npy"
np.save(freq_pnt_file.format(baseline_name), np.array([freqs, average_recall_freq]))

recall1000_pnt_file = "./baseline_pnt_pairs/{:s}-newtst_recall_pnt_pairs.npy"

np.save(recall1000_pnt_file.format(baseline_name), np.array([tiou_thresholds, recall]))

recall_freq_pnt_file = "./baseline_pnt_pairs/{:s}-newtst_recall_freq_pnt_pairs.npy"
np.save(recall_freq_pnt_file.format(baseline_name), np.array([tiou_thresholds_freq, recall_freq]))

