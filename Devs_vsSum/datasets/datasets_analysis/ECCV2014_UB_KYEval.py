import os
import sys

import numpy as np

import vsSummDevs.datasets.SumMe.path_vars as dataset_pathvars

project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import vsSummDevs.datasets.getShotBoundariesECCV2016 as getSegs
from vsSummDevs.datasets.SumMe import SumMeMultiViewFeatureLoader

video_file_stem_names = dataset_pathvars.file_names
video_file_stem_names.sort()
pdefined_segs = getSegs.getSumMeShotBoundaris()
sample_rate = 5

# video_file_stem_names.sort()
from vsSummDevs.SumEvaluation import vsum_tools, rep_conversions

t_acc = 0
for video_idx, s_filename in enumerate(video_file_stem_names):
    _, user_labels, feature_sizes = SumMeMultiViewFeatureLoader.load_by_name(s_filename, doSoftmax=False)
    s_seg = pdefined_segs[s_filename]
    avg_labels = np.mean(user_labels, axis=1)
    nframes = user_labels.shape[0]
    intevals = rep_conversions.convert_seg2interval(s_seg, nframes)
    intevals = np.asarray(intevals)
    nfps = []
    for i in range(1, len(s_seg)):
        nfps.append(s_seg[i] - s_seg[i-1])

    generated_summary = vsum_tools.generate_summary(avg_labels, intevals, nframes, nfps, positions=np.asarray(range(nframes)))

    my_generated_summary =  rep_conversions.framescore2frame01score(avg_labels, s_seg)

    s_F1_score, _, _ = vsum_tools.evaluate_summary(generated_summary, user_labels.transpose(), eval_metric='max')

    t_acc += (s_F1_score)
print "Total MinAcc: {:.04f}".format(t_acc / len(video_file_stem_names))



