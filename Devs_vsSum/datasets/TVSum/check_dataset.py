import os
import h5py
import LoadLabels
from vsSummDevs.datasets import getShotBoundariesECCV2016
from vsSummDevs.SumEvaluation import rep_conversions

dataset_name = 'tvsum'
tvsum_gt = LoadLabels.load_annotations()
tv_segments = getShotBoundariesECCV2016.getTVSumShotBoundaris()


KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
h5f_path = os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(dataset_name))
dataset = h5py.File(h5f_path, 'r')
dataset_keys = dataset.keys()
n_videos = len(dataset_keys)

index_dict = getShotBoundariesECCV2016.getTVSumCorrespondecesKZ()

for s_videostem in tvsum_gt.keys():
    user_annotations = tvsum_gt[s_videostem]
    s_segments = tv_segments[s_videostem]
    # key = dataset_keys[index_dict[s_videostem]]
    key = 'video_{:d}'.format(index_dict[s_videostem]+1)
    s_user_score = user_annotations['video_user_scores'][:,0]
    s_user_score01 = rep_conversions.framescore2frame01score(s_user_score.tolist(), s_segments.tolist())
    ky_user_summary = dataset[key]['user_summary'][...]
    ky_user_summary0 = ky_user_summary[0]
    cps = dataset[key]['change_points'][...]
    print "DEBUG"





print "DEBUG"