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
import ActionLocalizationDevs.PropEval.prop_eval_utils

# Define plot style.
# for file_idx in range(1, 300, 10):
file_idx = 71
Freq = 0.2
new_method = 'lstm2heads_{:04d}_fix_t'.format(file_idx)

method = {
          'TURN-C3D-16': {'legend': 'TURN-C3D-16',
                         'color': np.array([224, 44, 119]) / 255.0,
                         'marker': None,
                         'linewidth': 6.5,
                         'linestyle': '-'},

          'TURN-FLOW-16': {'legend': 'TURN-FLOW-16',
                          'color': np.array([224, 224, 119]) / 255.0,
                          'marker': None,
                          'linewidth': 6.5,
                          'linestyle': '-'},
          new_method: {'legend': new_method,
                           'color': np.array([0, 0, 254]) / 255.0,
                           'marker': None,
                           'linewidth': 6.5,
                           'linestyle': '-'}
          }

fn_size = 30
legend_size = 27.5


recall_freq_pnt_pairs = {}

recall_freq_pnt_pairs['TURN-C3D-16'] = np.load("./ref_pnt_pairs/TURN-C3D-16_{:.2f}_recall_freq_pnt_pairs.npy".format(Freq))
recall_freq_pnt_pairs['TURN-FLOW-16'] = np.load("./ref_pnt_pairs/TURN-FLOW-16_{:.2f}_recall_freq_pnt_pairs.npy".format(Freq))
recall_freq_pnt_pairs[new_method] = np.load("./ref_pnt_pairs/{:s}_{:.2f}_recall_freq_pnt_pairs.npy".format(new_method, Freq))

legends = ['TURN-C3D-16', 'TURN-FLOW-16', new_method]
# legends = ['DAPs','SCNN-prop','TURN-AP']
# legends = ['TURN-AP']

plt.figure(num=None, figsize=(12, 10))


plt.figure(num=None, figsize=(12, 10))
# Plots Average Recall vs Average number of proposals.
for _key in legends:
    plt.plot(recall_freq_pnt_pairs[_key][0, :], recall_freq_pnt_pairs[_key][1, :],
             label=method[_key]['legend'],
             color=method[_key]['color'],
             linewidth=method[_key]['linewidth'],
             linestyle=str(method[_key]['linestyle']),
             marker=str(method[_key]['marker']))

plt.ylabel('Recall@F={:.2f}'.format(Freq), fontsize=fn_size)
plt.xlabel('tIoU', fontsize=fn_size)
plt.grid(b=True, which="both")
plt.ylim([0, 1])
plt.xlim([0.1, 1])
plt.xticks(np.arange(0.0, 1.1, 0.2))
plt.legend(legends, loc=3, prop={'size': legend_size})
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
plt.savefig("./results/{:s}_recall_freq_{:.2f}.png".format(new_method, Freq), bbox_inches="tight")
# plt.show()
