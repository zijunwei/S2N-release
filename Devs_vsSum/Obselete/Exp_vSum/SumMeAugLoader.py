import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import datasets.SumMe.BNInceptionFeatureLoader as SumMeLoader
import datasets.SumMe.path_vars as SumMe_paths
import datasets.TVSum.BNInceptionFeatureLoader as TVSumLoader
import datasets.TVSum.path_vars as TVSum_paths
import torch.utils.data as data
from Devs_vsSum.datasets import LoaderUtils
from datasets import KyLoader
import numpy as np
import torch
import progressbar

user_root = os.path.expanduser('~')


KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
data_loaders = {'summe': SumMeLoader, 'tvsum': TVSumLoader}

class Dataset(data.Dataset):
    def __init__(self, dataset_name='SumMe',split='train', clip_size=50, output3=False):
        if dataset_name.lower() not in ['summe', 'tvsum']:
            print('Unrecognized dataset {:s}'.format(dataset_name))
            sys.exit(-1)
        self.dataset_name = dataset_name
        self.split = split
        self.filenames = datasetpaths[self.dataset_name.lower()].file_names
        self.data_loader = data_loaders[self.dataset_name.lower()]

        n_files = len(self.filenames)
        if self.split == 'train':
            self.filenames = self.filenames[:int(0.8*n_files)]
        elif self.split == 'val':
            self.filenames = self.filenames[int(0.8*n_files):]
        else:
            print("Unrecognized split:{:s}".format(self.split))
            sys.exit(-1)
        self.clip_size = clip_size
        self.output3 = output3

        Kydataset = KyLoader.loadKyDataset(self.dataset_name.lower())
        conversion = KyLoader.loadConversion(self.dataset_name.lower())
        self.raw2Ky = conversion[0]
        self.Ky2raw = conversion[1]


        self.features = {}
        self.avg_summary = {}
        self.annotations = []
        pbar = progressbar.ProgressBar(max_value=len(self.filenames))
        print("Processing {:s}\t{:s} data".format(self.dataset_name, self.split))
        for file_dix, s_filename in enumerate(self.filenames):
            pbar.update(file_dix)
            Kykey = self.raw2Ky[s_filename]
            s_features, _, _ = self.data_loader.load_by_name(s_filename)
            s_usersummaries = Kydataset[Kykey]['user_summary'][...]
            s_usersummaries = s_usersummaries.transpose()

            self.features[s_filename] = s_features
            s_avg_summary = np.mean(s_usersummaries, axis=1)
            self.avg_summary[s_filename] = s_avg_summary
            s_video_len = len(s_avg_summary)

            s_segments = LoaderUtils.convertlabels2segs(s_usersummaries)
            for s_seg in s_segments:
                sSegment = LoaderUtils.Segment(s_filename)
                s_score  = np.mean(s_avg_summary[s_seg[0]:s_seg[1]])
                sSegment.initId(s_seg[0], s_seg[1], s_video_len, score=s_score)
                self.annotations.append(sSegment)

        self.augmented_ratio = 10

        self.dataset_size = len(self.annotations)
        print("{:s}\t{:s}\t{:d}".format(self.dataset_name, self.split, self.dataset_size))

    def __getitem__(self, index):

        s_annotation = self.annotations[index/self.augmented_ratio]
        frame_offset = self.clip_size/2/self.augmented_ratio * (index % self.augmented_ratio)
        frame_start = max(s_annotation.startIdx - frame_offset, 0)
        frame_end = frame_start + self.clip_size
        if frame_end > s_annotation.seq_length:
            offset = frame_end - s_annotation.seq_length
            frame_start -= offset
            frame_end = s_annotation.seq_length

        s_groundtruth = torch.LongTensor(
            [s_annotation.startIdx - frame_start, min(s_annotation.endIdx - frame_start, self.clip_size - 1)])
        s_feature = torch.from_numpy(self.features[s_annotation.video_stem][frame_start:frame_end]).float()

        if self.output3:
            s_score = torch.FloatTensor([float(s_annotation.score)])
            return s_feature, s_groundtruth, s_score
        return s_feature, s_groundtruth

    def __len__(self):
        return self.dataset_size * self.augmented_ratio  # Predefine size...

if __name__ == '__main__':

    sDataset = Dataset(dataset_name='TVSum', split='train')

    # HicoLoader = torch.utils.data.DataLoader(HicoDataset, batch_size=20, shuffle=False)
    for i, (image, label) in enumerate(sDataset):
        pass

        # print "DEBUG"