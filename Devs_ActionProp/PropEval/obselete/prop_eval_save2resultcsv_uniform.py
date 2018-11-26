import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import matplotlib as mpl

mpl.use('Agg')
import numpy as np
import pandas as pd
import pickle
from ActionLocalizationDevs.PropEval.obselete import prop_eval_utils_mod as prop_eval_utils

baseline_name = 'scnnprop_loc'

prediction_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/baselines_results/{:s}_thumos14_test.csv'.format(baseline_name)
ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'
frm_nums = pickle.load(open("./frm_num.pkl"))


daps_results = pd.read_csv(prediction_file, sep=' ')
ground_truth = pd.read_csv(ground_truth_file, sep=' ')
average_recall, average_nr_proposals = prop_eval_utils.average_recall_vs_nr_proposals(daps_results,
                                                                      ground_truth)
average_recall_freq, freqs = prop_eval_utils.average_recall_vs_freq(daps_results, ground_truth, frm_nums)

# Computes recall for different tiou thresholds at a fixed average number of proposals.
recall, tiou_thresholds = prop_eval_utils.recall_vs_tiou_thresholds(daps_results, ground_truth,
                                                    nr_proposals=1000)

recall_freq, tiou_thresholds_freq = prop_eval_utils.recall_freq_vs_tiou_thresholds(daps_results, ground_truth, frm_nums,
                                                                                   pdefined_freq=1.0)



avg_prop_pnt_file = "./baseline_pnt_pairs/{:s}_avg_prop_pnt_pairs.npy"
np.save(avg_prop_pnt_file.format(baseline_name), np.array([average_nr_proposals, average_recall]))

freq_pnt_file = "./baseline_pnt_pairs/{:s}_freq_pnt_pairs.npy"
np.save(freq_pnt_file.format(baseline_name), np.array([freqs, average_recall_freq]))

recall1000_pnt_file = "./baseline_pnt_pairs/{:s}_recall_pnt_pairs.npy"

np.save(recall1000_pnt_file.format(baseline_name), np.array([tiou_thresholds, recall]))

recall_freq_pnt_file = "./baseline_pnt_pairs/{:s}_recall_freq_pnt_pairs.npy"
np.save(recall_freq_pnt_file.format(baseline_name), np.array([tiou_thresholds_freq, recall_freq]))


