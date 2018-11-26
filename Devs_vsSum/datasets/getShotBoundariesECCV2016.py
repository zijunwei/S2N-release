import os
import sys
import scipy.io
import Devs_vsSum.datasets.SumMe.path_vars
import Devs_vsSum.datasets.TVSum.path_vars
import glob
import shutil


def getSumMeShotBoundaris():
    groundtruth_paths = glob.glob(os.path.join(Devs_vsSum.datasets.SumMe.path_vars.dataset_dir, 'frames', '*/'))
    groundtruth_paths.sort()
    shot_file = os.path.join(Devs_vsSum.datasets.SumMe.path_vars.dataset_dir, 'shot_SumMe.mat')
    shotfile_content = scipy.io.loadmat(shot_file)
    assert len(shotfile_content['shot_boundaries']) == len(groundtruth_paths)
    video_shots = {}
    for idx, (shots, s_path) in enumerate(zip(shotfile_content['shot_boundaries'], groundtruth_paths)):

        n_files = len(glob.glob(os.path.join(s_path, '*.jpg')))
        shots = shots[0].squeeze(0)
        if n_files < shots[-1]:
            print "Inconsistenent Frame {:d} vs {:d} for {:s}".format(shots[-1], n_files, (s_path))
            sys.exit(-1)
        video_stem = s_path.split(os.sep)[-2].replace(' ', '_')
        video_shots[video_stem] = shots
    return video_shots






def getTVSumShotBoundaris():
    groundtruth_paths = glob.glob(os.path.join(Devs_vsSum.datasets.TVSum.path_vars.dataset_dir, 'frames', '*/'))
    groundtruth_paths.sort()
    # groundtruth_paths = groundtruth_paths[::-1]
    shot_file = os.path.join(Devs_vsSum.datasets.TVSum.path_vars.dataset_dir, 'shot_TVSum.mat')
    shotfile_content = scipy.io.loadmat(shot_file)
    assert len(shotfile_content['shot_boundaries']) == len(groundtruth_paths)
    video_shots = {}

    shot_nframes = []
    for shots in shotfile_content['shot_boundaries']:
        n_files = shots[0].squeeze(0)[-1]
        shot_nframes.append(n_files)

    video_frames = {}

    for s_path in groundtruth_paths:
        n_files = len(glob.glob(os.path.join(s_path, '*.jpg')))
        video_frames[s_path.split(os.sep)[-2]] =  n_files

    shot_dict = {}
    for idx,  s_shot_frame in enumerate(shot_nframes):
        for s_videoframe in video_frames.keys():
            if abs(s_shot_frame - video_frames[s_videoframe])<=2:
                shot_dict[s_videoframe] = idx

    for idx, s_path in enumerate(groundtruth_paths):
        s_videostem = s_path.split(os.sep)[-2]
        n_files = len(glob.glob(os.path.join(s_path, '*.jpg')))
        shots = shotfile_content['shot_boundaries'][shot_dict[s_videostem]][0].squeeze(0)
        shots[-1] = n_files
        if n_files < shots[-1]:
            print "Inconsistenent Frame {:d} vs {:d} for {:s}".format(shots[-1], n_files, (s_path))
            sys.exit(-1)
        video_stem = s_path.split(os.sep)[-2].replace(' ', '_')
        video_shots[video_stem] = shots
    return video_shots


def getTVSumCorrespondecesKZ():
    groundtruth_paths = glob.glob(os.path.join(Devs_vsSum.datasets.TVSum.path_vars.dataset_dir, 'frames', '*/'))
    groundtruth_paths.sort()
    # groundtruth_paths = groundtruth_paths[::-1]
    shot_file = os.path.join(Devs_vsSum.datasets.TVSum.path_vars.dataset_dir, 'shot_TVSum.mat')
    shotfile_content = scipy.io.loadmat(shot_file)
    assert len(shotfile_content['shot_boundaries']) == len(groundtruth_paths)
    video_shots = {}

    shot_nframes = []
    for shots in shotfile_content['shot_boundaries']:
        n_files = shots[0].squeeze(0)[-1]
        shot_nframes.append(n_files)

    video_frames = {}

    for s_path in groundtruth_paths:
        n_files = len(glob.glob(os.path.join(s_path, '*.jpg')))
        video_frames[s_path.split(os.sep)[-2]] =  n_files

    shot_dict = {}
    for idx,  s_shot_frame in enumerate(shot_nframes):
        for s_videoframe in video_frames.keys():
            if abs(s_shot_frame - video_frames[s_videoframe])<=2:
                shot_dict[s_videoframe] = idx

    return shot_dict

if __name__ == '__main__':
    # groundtruth_paths = glob.glob(os.path.join(datasets.SumMe.path_vars.dataset_dir, 'frames', '*/'))
    # groundtruth_paths.sort()
    # shot_file = os.path.join(datasets.SumMe.path_vars.dataset_dir, 'shot_SumMe.mat')
    # shotfile_content = scipy.io.loadmat(shot_file)
    # assert len(shotfile_content['shot_boundaries']) == len(groundtruth_paths)
    # video_shots = {}
    # for idx, (shots, s_path) in enumerate(zip(shotfile_content['shot_boundaries'], groundtruth_paths)):
    #
    #     n_files = len(glob.glob(os.path.join(s_path, '*.jpg')))
    #     shots = shots[0].squeeze(0)
    #     if n_files < shots[-1]:
    #         print "Inconsistenent Frame {:d} vs {:d} for {:s}".format(shots[-1], n_files, (s_path))
    #         sys.exit(-1)
    #     video_stem = s_path.split(os.sep)[-2].replace(' ', '_')
    #     video_shots[video_stem] = shots

    segments = getTVSumShotBoundaris()
    print "DEBUG"
