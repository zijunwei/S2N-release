import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)
import matplotlib as mpl

mpl.use('Agg')
import pandas as pd
import pickle
import progressbar


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


save_name = 'lstm2heads_0071_fix_t'
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
ground_truth = pd.read_csv(ground_truth_file, sep=' ')

Freq_pred = 0
Freq_gt = 0
for s_video_name in frm_nums.keys():
    s_n_frames = frm_nums[s_video_name]
    dap_idxes = daps_results['video-name'] == s_video_name
    gt_idxes = ground_truth['video-name'] == s_video_name
    gt_proposals = ground_truth[gt_idxes]
    dap_proposals = daps_results[dap_idxes]
    s_n_proposals = len(dap_proposals)
    s_n_gts = len(gt_proposals)
    Freq_pred += s_n_proposals*1./s_n_frames * 30.
    Freq_gt += s_n_gts*1./s_n_frames * 30.
Freq_pred = Freq_pred*1. / len(frm_nums)
Freq_gt = Freq_gt*1./len(frm_nums)
print("Freq of {:s}\t{:.4f}".format(save_name, Freq_pred))
print("DB")

# Computes average recall vs average number of proposals.





