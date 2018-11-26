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

# Define plot style.
# for file_idx in range(1, 300, 10):
# file_idx = 71
new_method = 'Ours'

method = {'DAPs': {'legend': 'DAPs-prop',
                   'color': np.array([102, 166, 30]) / 255.0,
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
                     'color': np.array([0, 0, 0]) / 255.0,
                     'marker': None,
                     'linewidth': 6.5,
                     'linestyle': '-'},
          'TURN-C3D': {'legend': 'TURN-C3D',
                          'color': np.array([224, 44, 119]) / 255.0,
                          'marker': None,
                          'linewidth': 6.5,
                          'linestyle': '-'},
          'TURN-FLOW': {'legend': 'TURN-FLOW',
                           'color': np.array([224, 224, 119]) / 255.0,
                           'marker': None,
                           'linewidth': 6.5,
                           'linestyle': '-'},

          new_method: {'legend': 'Ours-C3D',
                           'color': np.array([0, 0, 254]) / 255.0,
                           'marker': None,
                           'linewidth': 6.5,
                           'linestyle': '-'}
          }

fn_size = 30
legend_size = 15

# reference points load:
avg_prop_pnt_pairs = {}
avg_prop_pnt_pairs['DAPs'] = np.load("./baseline_pnt_pairs/daps_avg_prop_pnt_pairs.npy")
# avg_prop_pnt_pairs['SCNN-prop'] = np.load("./baseline_pnt_pairs/scnnprop_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Sparse-prop'] = np.load("./baseline_pnt_pairs/sparseprop_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Sliding Window'] = np.load("./baseline_pnt_pairs/slidingwindow_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['Random'] = np.load("./baseline_pnt_pairs/random_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['TURN-C3D'] = np.load("./baseline_pnt_pairs/TURN-C3D_avg_prop_pnt_pairs.npy")
avg_prop_pnt_pairs['TURN-FLOW'] = np.load("./baseline_pnt_pairs/TURN-FLOW_avg_prop_pnt_pairs.npy")

avg_prop_pnt_pairs[new_method] = np.load("./baseline_pnt_pairs/{:s}_avg_prop_pnt_pairs.npy".format(new_method))

freq_pnt_pairs = {}
freq_pnt_pairs['DAPs'] = np.load("./baseline_pnt_pairs/daps_freq_pnt_pairs.npy")
# freq_pnt_pairs['SCNN-prop'] = np.load("./baseline_pnt_pairs/scnnprop_freq_pnt_pairs.npy")
freq_pnt_pairs['Sparse-prop'] = np.load("./baseline_pnt_pairs/sparseprop_freq_pnt_pairs.npy")
freq_pnt_pairs['Sliding Window'] = np.load("./baseline_pnt_pairs/slidingwindow_freq_pnt_pairs.npy")
freq_pnt_pairs['Random'] = np.load("./baseline_pnt_pairs/random_freq_pnt_pairs.npy")
freq_pnt_pairs['TURN-C3D'] = np.load("./baseline_pnt_pairs/TURN-C3D_freq_pnt_pairs.npy")
freq_pnt_pairs['TURN-FLOW'] = np.load("./baseline_pnt_pairs/TURN-FLOW_freq_pnt_pairs.npy")

freq_pnt_pairs[new_method] = np.load("./baseline_pnt_pairs/{:s}_freq_pnt_pairs.npy".format(new_method))

recall1000_pnt_pairs = {}
recall1000_pnt_pairs['DAPs'] = np.load("./baseline_pnt_pairs/daps_recall_pnt_pairs.npy")
# recall1000_pnt_pairs['SCNN-prop'] = np.load("./baseline_pnt_pairs/scnnprop_recall_pnt_pairs.npy")
recall1000_pnt_pairs['Sparse-prop'] = np.load("./baseline_pnt_pairs/sparseprop_recall_pnt_pairs.npy")
recall1000_pnt_pairs['Sliding Window'] = np.load("./baseline_pnt_pairs/slidingwindow_recall_pnt_pairs.npy")
recall1000_pnt_pairs['Random'] = np.load("./baseline_pnt_pairs/random_recall_pnt_pairs.npy")
recall1000_pnt_pairs['TURN-C3D'] = np.load("./baseline_pnt_pairs/TURN-C3D_recall_pnt_pairs.npy")
recall1000_pnt_pairs['TURN-FLOW'] = np.load("./baseline_pnt_pairs/TURN-FLOW_recall_pnt_pairs.npy")

recall1000_pnt_pairs[new_method] = np.load("./baseline_pnt_pairs/{:s}_recall_pnt_pairs.npy".format(new_method))

recall_freq_pnt_pairs = {}
recall_freq_pnt_pairs['DAPs'] = np.load("./baseline_pnt_pairs/daps_recall_freq_pnt_pairs.npy")
# recall_freq_pnt_pairs['SCNN-prop'] = np.load("./baseline_pnt_pairs/scnnprop_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['Sparse-prop'] = np.load("./baseline_pnt_pairs/sparseprop_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['Sliding Window'] = np.load("./baseline_pnt_pairs/slidingwindow_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['Random'] = np.load("./baseline_pnt_pairs/random_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs[new_method] = np.load("./baseline_pnt_pairs/{:s}_recall_freq_pnt_pairs.npy".format(new_method))
recall_freq_pnt_pairs['TURN-C3D'] = np.load("./baseline_pnt_pairs/TURN-C3D_recall_freq_pnt_pairs.npy")
recall_freq_pnt_pairs['TURN-FLOW'] = np.load("./baseline_pnt_pairs/TURN-FLOW_recall_freq_pnt_pairs.npy")
legends = ['Random', 'Sliding Window', 'Sparse-prop', 'DAPs', 'TURN-C3D', 'TURN-FLOW', new_method]

plt.figure(num=None, figsize=(12, 10))

# Plots Average Recall vs Average number of proposals.
for _key in legends:
    upper_threshold = 300
    x_indices = avg_prop_pnt_pairs[_key][0, :]
    selected_index = 0
    for idx, x in enumerate(x_indices):
        if x < upper_threshold:
            selected_index = idx


    plt.semilogx(avg_prop_pnt_pairs[_key][0, :selected_index], avg_prop_pnt_pairs[_key][1, :selected_index],
                 label=method[_key]['legend'],
                 color=method[_key]['color'],
                 linewidth=method[_key]['linewidth'],
                 linestyle=str(method[_key]['linestyle']),
                 marker=str(method[_key]['marker']))

plt.ylabel('Average Recall', fontsize=fn_size)
plt.xlabel('Average number of retrieved proposals', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 0.6])
plt.xlim([10 ** 1, 3.5 * 10 ** 2])
plt.yticks(np.arange(0.0, 0.7, 0.2))
plt.legend(legends, loc=2, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig("./results/{:s}_avg_recall_pub.pdf".format(new_method), bbox_inches="tight")
# plt.show()

plt.figure(num=None, figsize=(12, 10))
# Plots Average Recall vs Average number of proposals.
for _key in legends:
    upper_threshold = 1.3
    x_indices = freq_pnt_pairs[_key][0, :]
    selected_index = 0
    for idx, x in enumerate(x_indices):
        if x < upper_threshold:
            selected_index = idx
    plt.semilogx(freq_pnt_pairs[_key][0, :selected_index], freq_pnt_pairs[_key][1, :selected_index],
                 label=method[_key]['legend'],
                 color=method[_key]['color'],
                 linewidth=method[_key]['linewidth'],
                 linestyle=str(method[_key]['linestyle']),
                 marker=str(method[_key]['marker']))

plt.ylabel('Average Recall', fontsize=fn_size)
plt.xlabel('Proposal frequency', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 0.6])
plt.xlim([10 ** (-1), 1.2])
plt.yticks(np.arange(0.0, 0.6, 0.2))
plt.legend(legends, loc=2, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig("./results/{:s}_freq_pub.pdf".format(new_method), bbox_inches="tight")
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
plt.savefig("./results/{:s}_recall1000_pub.pdf".format(new_method), bbox_inches="tight")
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
plt.savefig("./results/{:s}_recall_freq_pub.pdf".format(new_method), bbox_inches="tight")
# plt.show()
