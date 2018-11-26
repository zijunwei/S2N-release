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


save_name = 'TURN-FLOW-16'
prediction_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/pkl_files/{:s}_thumos14.pkl'.format(save_name)
ground_truth_file = '/home/zwei/Dev/NetModules/ActionLocalizationDevs/PropEval/thumos14_test_groundtruth.csv'
frm_nums = pickle.load(open("./frm_num.pkl"))
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

# Define plot style.
method = {'DAPs': {'legend': 'DAPs-prop',
                   'color': np.array([102, 166, 30]) / 255.0,
                   'marker': None,
                   'linewidth': 6.5,
                   'linestyle': '-'},
          'SCNN-prop': {'legend': 'SCNN-prop',
                        'color': np.array([230, 171, 2]) / 255.0,
                        'marker': None,
                        'linewidth': 6.5,
                        'linestyle': '-'},
          'Sparse-prop': {'legend': 'Sparse-prop',
                          'color': np.array([153, 78, 160]) / 255.0,
                          'marker': None,
                          'linewidth': 6.5,
                          'linestyle': '-'},
          'Sliding Window': {'legend': 'Sliding Window',
                             'color': np.array([205, 110, 51]) / 255.0,
                             'marker': None,
                             'linewidth': 6.5,
                             'linestyle': '-'},
          'Random': {'legend': 'Random',
                     'color': np.array([132, 132, 132]) / 255.0,
                     'marker': None,
                     'linewidth': 6.5,
                     'linestyle': '-'},
          'TURN-AP': {'legend': 'TURN-AP',
                         'color': np.array([224, 44, 119]) / 255.0,
                         'marker': None,
                         'linewidth': 6.5,
                         'linestyle': '-'}
          }

fn_size = 30
legend_size = 27.5

# reference points load:
avg_prop_pnt_pairs = {}
avg_prop_pnt_pairs['DAPs'] = np.load("./ref_pnt_pairs/DAP_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['SCNN-prop'] = np.load("./ref_pnt_pairs/SCNN_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Sparse-prop'] = np.load("./ref_pnt_pairs/sparse_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Sparse-prop'] = avg_prop_pnt_pairs['Sparse-prop']
avg_prop_pnt_pairs['flow'] = np.load("./ref_pnt_pairs/flow_svm_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Sliding Window'] = np.load("./ref_pnt_pairs/sliding_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Random'] = np.load("./ref_pnt_pairs/random_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['TURN-AP'] = np.array([average_nr_proposals, average_recall])

freq_pnt_pairs = {}
freq_pnt_pairs['DAPs'] = np.load("./ref_pnt_pairs/DAP_freq_pnt_pairs.npy")
freq_pnt_pairs['SCNN-prop'] = np.load("./ref_pnt_pairs/scnn_freq_pnt_pairs.npy")
freq_pnt_pairs['Sparse-prop'] = np.load("./ref_pnt_pairs/sparse_freq_pnt_pairs.npy")
freq_pnt_pairs['Sparse-prop'] = freq_pnt_pairs['Sparse-prop'][:, 0:-2]
freq_pnt_pairs['flow'] = np.load("./ref_pnt_pairs/flow_svm_freq_pnt_pairs.npy")
freq_pnt_pairs['Sliding Window'] = np.load("./ref_pnt_pairs/sliding_freq_pnt_pairs.npy")
freq_pnt_pairs['Random'] = np.load("./ref_pnt_pairs/random_freq_pnt_pairs.npy")
freq_pnt_pairs['TURN-AP'] = np.array([freqs, average_recall_freq])

recall1000_pnt_pairs = {}
recall1000_pnt_pairs['DAPs'] = np.load("./ref_pnt_pairs/DAP_recall_pnt_pairs.npy")
recall1000_pnt_pairs['SCNN-prop'] = np.load("./ref_pnt_pairs/SCNN_recall_pnt_pairs.npy")
recall1000_pnt_pairs['Sparse-prop'] = np.load("./ref_pnt_pairs/sparse_recall_pnt_pairs.npy")
recall1000_pnt_pairs['flow'] = np.load("./ref_pnt_pairs/flow_svm_recall_pnt_pairs.npy")
recall1000_pnt_pairs['Sliding Window'] = np.load("./ref_pnt_pairs/sliding_recall_pnt_pairs.npy")
recall1000_pnt_pairs['Random'] = np.load("./ref_pnt_pairs/random_recall_pnt_pairs.npy")
recall1000_pnt_pairs['TURN-AP'] = np.array([tiou_thresholds, recall])

recall_freq_pnt_pairs = {}
recall_freq_pnt_pairs['DAPs'] = np.load("./ref_pnt_pairs/DAPs_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['SCNN-prop'] = np.load("./ref_pnt_pairs/scnn_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['Sparse-prop'] = np.load("./ref_pnt_pairs/sparse_recall_freq_pnt_pairs.npy")
# recall_freq_pnt_pairs['flow'] = np.load("./ref_pnt_pairs/flow_svm_recall_pnt_pairs.npy")
recall_freq_pnt_pairs['Sliding Window'] = np.load("./ref_pnt_pairs/sliding_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['Random'] = np.load("./ref_pnt_pairs/random_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['TURN-AP'] = np.array([tiou_thresholds_freq, recall_freq])

legends = ['Random', 'Sliding Window', 'Sparse-prop', 'DAPs', 'SCNN-prop', 'TURN-AP']
# legends = ['DAPs','SCNN-prop','TURN-AP']
# legends = ['TURN-AP']

plt.figure(num=None, figsize=(12, 10))

# Plots Average Recall vs Average number of proposals.
for _key in legends:
    plt.semilogx(avg_prop_pnt_pairs[_key][0, :], avg_prop_pnt_pairs[_key][1, :],
                 label=method[_key]['legend'],
                 color=method[_key]['color'],
                 linewidth=method[_key]['linewidth'],
                 linestyle=str(method[_key]['linestyle']),
                 marker=str(method[_key]['marker']))

plt.ylabel('Average Recall', fontsize=fn_size)
plt.xlabel('Average number of retrieved proposals', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 0.8])
plt.xlim([10 ** 1, 5 * 10 ** 3])
plt.yticks(np.arange(0.0, 0.9, 0.2))
plt.legend(legends, loc=2, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig(os.path.basename(prediction_file).split(".pkl")[0] + "_avg_recall.pdf", bbox_inches="tight")
# plt.show()

plt.figure(num=None, figsize=(12, 10))
# Plots Average Recall vs Average number of proposals.
for _key in legends:
    plt.semilogx(freq_pnt_pairs[_key][0, :], freq_pnt_pairs[_key][1, :],
                 label=method[_key]['legend'],
                 color=method[_key]['color'],
                 linewidth=method[_key]['linewidth'],
                 linestyle=str(method[_key]['linestyle']),
                 marker=str(method[_key]['marker']))

plt.ylabel('Average Recall', fontsize=fn_size)
plt.xlabel('Proposal frequency', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 0.8])
plt.xlim([10 ** (-1), 10])
plt.yticks(np.arange(0.0, 0.9, 0.2))
plt.legend(legends, loc=2, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig(os.path.basename(prediction_file).split(".pkl")[0] + "_freq.pdf", bbox_inches="tight")
# plt.show()

# Plots recall at different tiou thresholds.
plt.figure(num=None, figsize=(12, 10))
for _key in legends:
    plt.plot(recall1000_pnt_pairs[_key][0, :], recall1000_pnt_pairs[_key][1, :],
             label=method[_key]['legend'],
             color=method[_key]['color'],
             linewidth=method[_key]['linewidth'],
             linestyle=str(method[_key]['linestyle']),
             marker=str(method[_key]['marker']))

plt.ylabel('Recall@AN=1000', fontsize=fn_size)
plt.xlabel('tIoU', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 1])
plt.xlim([0.1, 1])
plt.xticks(np.arange(0.0, 1.1, 0.2))
plt.legend(legends, loc=3, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig(os.path.basename(prediction_file).split(".pkl")[0] + "_recall1000.pdf", bbox_inches="tight")
# plt.show()

plt.figure(num=None, figsize=(12, 10))
# Plots Average Recall vs Average number of proposals.
for _key in legends:
    plt.plot(recall_freq_pnt_pairs[_key][0, :], recall_freq_pnt_pairs[_key][1, :],
             label=method[_key]['legend'],
             color=method[_key]['color'],
             linewidth=method[_key]['linewidth'],
             linestyle=str(method[_key]['linestyle']),
             marker=str(method[_key]['marker']))

plt.ylabel('Recall@F=1.0', fontsize=fn_size)
plt.xlabel('tIoU', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 1])
plt.xlim([0.1, 1])
plt.xticks(np.arange(0.0, 1.1, 0.2))
plt.legend(legends, loc=3, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig(os.path.basename(prediction_file).split(".pkl")[0] + "_recall_freq.pdf", bbox_inches="tight")
# plt.show()
