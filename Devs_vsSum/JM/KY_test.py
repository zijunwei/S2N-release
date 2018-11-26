

import numpy as np
import os, h5py, vsSummDevs.SumEvaluation.vsum_tools


def createPositions(nFrames, framerate):
    positions = range(0, nFrames, framerate)
    positions = np.asarray(positions)

    return positions

def test():

    eval_datasets = ['summe', 'tvsum']
    eval_dataset = eval_datasets[1]


    if eval_dataset == 'summe':
        eval_metric = 'max'
    elif eval_dataset == 'tvsum':
        eval_metric = 'avg'


    KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
    h5f_path = os.path.join(KY_dataset_path, 'eccv16_dataset_{:s}_google_pool5.h5'.format(eval_dataset))
    dataset = h5py.File(h5f_path, 'r')
    dataset_keys = dataset.keys()
    n_videos = len(dataset_keys)

    fms = []
    precs = []
    recs = []
    frame_rate = 15 # this is the best possible frame rate
    for i_video in range(n_videos):
        key = dataset_keys[i_video]
        data_x = dataset[key]['features'][...]

        cps = dataset[key]['change_points'][...]
        n_frames = dataset[key]['n_frames'][()]
        nfps = dataset[key]['n_frame_per_seg'][...].tolist()
        # positions = dataset[key]['picks'][...]
        positions = createPositions(n_frames, frame_rate)
        user_summary = dataset[key]['user_summary'][...]
        avg_summary = np.mean(user_summary, axis=0)
        probs = avg_summary[positions]
        machine_summary = vsSummDevs.SumEvaluation.vsum_tools.generate_summary(probs, cps, n_frames, nfps, positions)
        fm,prec,rec = vsSummDevs.SumEvaluation.vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
        fms.append(fm)
        precs.append(prec)
        recs.append(rec)

    mean_fm = np.mean(fms)

    print "Mean F1 Score: {:.04f}@FrameRate:{:d}".format(mean_fm, frame_rate)

    dataset.close()

if __name__ == '__main__':
    test()