import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import numpy as np
from vsSummDevs.datasets import KyLoader

import matplotlib.pyplot as plt
import seaborn as sns

def draw(user_summaries, cps):
    n_colors = user_summary.shape[0]
    n_frames = user_summary.shape[1]
    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", n_colors))
    palette = np.array(sns.cubehelix_palette(n_colors))
    s_cps = []
    for i in cps:
        s_cps.append(i[0])

    f = plt.figure(figsize=(8, 8))

    ax = plt.subplot()
    ax.set_ylim([0, 2])
    for i in range(n_colors):
        ax.plot(range(n_frames), user_summaries[i, :],c=palette[i])

    for i in s_cps:
        plt.axvline(x=i)




    return f


eval_dataset = 'TVSum'
dataset = KyLoader.loadKyDataset(eval_dataset)
video_frames = KyLoader.getKyVideoFrames(dataset)
dataset_keys =KyLoader.getKyDatasetKeys(dataset)

F1_scores = 0
frame_rate = 15

L2NormFeature=False
for video_idx, s_key in enumerate(dataset_keys):

    user_summary = dataset[s_key]['user_summary'][...]
    nfps = dataset[s_key]['n_frame_per_seg'][...].tolist()
    cps = dataset[s_key]['change_points'][...]
    n_frames = dataset[s_key]['n_frames'][()]
    positions = dataset[s_key]['picks'][...]

    f = draw(user_summary, cps)
    # f.show()
    save_name =  '{:s}.PNG'.format(s_key)
    plt.savefig(save_name)
print "overall F1 score: {:.04f}".format(F1_scores/len(dataset_keys))




# print "DEBUG"




