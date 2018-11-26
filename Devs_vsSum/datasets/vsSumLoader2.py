import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import h5py
import pickle as pkl
import Devs_vsSum.datasets
import Devs_vsSum.datasets.SumMe.BNInceptionFeatureLoader as SumMeLoader
import Devs_vsSum.datasets.SumMe.path_vars as SumMe_paths
import Devs_vsSum.datasets.TVSum.BNInceptionFeatureLoader as TVSumLoader
import Devs_vsSum.datasets.TVSum.path_vars as TVSum_paths
import progressbar
import torch.utils.data as data
import Devs_vsSum.Exp_vSum.LoaderUtils
from Devs_vsSum.datasets import KyLoader
import numpy as np
import torch

user_root = os.path.expanduser('~')


KY_dataset_path = os.path.join(os.path.expanduser('~'), 'datasets/KY_AAAI18/datasets')
datasetpaths = {'summe': SumMe_paths, 'tvsum': TVSum_paths}
data_loaders = {'summe': SumMeLoader, 'tvsum': TVSumLoader}

class Dataset(data.Dataset):
    def __init__(self, dataset_name='SumMe', split='train', clip_size=200, output_score=False, output_rdIdx=False, AugmentOffset=10, sample_rates=None):
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

        self.clip_size = clip_size
        self.output_score = output_score
        self.output_rdIdx = output_rdIdx
        # multi-scale summarization
        if sample_rates is None:
            self.sample_rates = [1]
        elif isinstance(sample_rates, list):
            self.sample_rates = sample_rates
        else:
            self.sample_rates = [sample_rates]


        Kydataset = KyLoader.loadKyDataset(self.dataset_name.lower())
        conversion = KyLoader.loadConversion(self.dataset_name.lower()) # the raw video names are renamed to video1...., we need to find the correcposndences
        self.raw2Ky = conversion[0]
        self.Ky2raw = conversion[1]

        self.features = {}
        self.avg_summary = {}
        self.data_instances = []
        print("Processing {:s}\t{:s} data".format(self.dataset_name, self.split))

        pbar = progressbar.ProgressBar(max_value=len(self.filenames))
        for file_dix, s_filename in enumerate(self.filenames):
            pbar.update(file_dix)
            Kykey = self.raw2Ky[s_filename]
            s_features, _, _ = self.data_loader.load_by_name(s_filename)
            s_usersummaries = Kydataset[Kykey]['user_summary'][...]
            s_usersummaries = s_usersummaries.transpose()

            self.features[s_filename] = s_features
            s_avg_summary = np.mean(s_usersummaries, axis=1)
            self.avg_summary[s_filename] = s_avg_summary
            n_frames = len(s_avg_summary)

            s_segments = vsSummDevs.Exp_vSum.LoaderUtils.convertlabels2segs(s_usersummaries) # load segments, check this function...
            #TODO: starting from here, you may consider changing it according to dataloader_c3dd_aug_fast
            for sample_rate in self.sample_rates:
                sample_rate_frames = int(n_frames/sample_rate)
                for s_seg in s_segments:
                    # s_score  = np.mean(s_avg_summary[s_seg[0]:s_seg[1]])

                    startIdx_l_range = max(0, int(s_seg[0]/sample_rate)+AugmentOffset - self.clip_size)
                    startIdx_r_range = min(sample_rate_frames, int(s_seg[1]/sample_rate) - AugmentOffset)

                    for s_startIdx_abs in range(startIdx_l_range, startIdx_r_range, AugmentOffset):
                        if s_startIdx_abs + self.clip_size >= sample_rate_frames:
                            break
                        s_segStartIdx_rel = max(0, int(s_seg[0]/sample_rate) - s_startIdx_abs)
                        s_segEndIdx_rel = min(int(s_seg[1]/sample_rate)- s_startIdx_abs, self.clip_size-1)
                        # if s_segEndIdx_rel < 0:
                        #     print "Debug"
                        # if s_segEndIdx_rel == self.clip_size:
                        #     print "DEBUG"
                        if s_segEndIdx_rel <= s_segStartIdx_rel+1:
                            # print "DEBUG"
                            continue

                        position_vector = range(s_startIdx_abs*sample_rate, (s_startIdx_abs+self.clip_size)*sample_rate, sample_rate)
                        sInstance = vsSummDevs.Exp_vSum.LoaderUtils.DataInstance(s_filename)
                        segment_summary = s_avg_summary[position_vector]
                        s_score = np.mean(segment_summary[s_segStartIdx_rel:s_segEndIdx_rel])
                        sInstance.initInstance(position_vector,s_segStartIdx_rel, s_segEndIdx_rel, s_score, self.clip_size)
                        if self.output_rdIdx:
                            rstart_idx, rend_idx = vsSummDevs.Exp_vSum.LoaderUtils.createRdIdxes(s_segStartIdx_rel, s_segEndIdx_rel, self.clip_size)
                            # if rstart_idx <0:
                            #     print "Debug"
                            # if rend_idx >= self.clip_size:
                            #     print "Debug"
                            # if rstart_idx >= rend_idx:
                            #     print "DEBUG"
                            r_score = np.mean(segment_summary[rstart_idx:rend_idx])
                            sInstance.addRdIdx(rstart_idx, rend_idx, r_score)

                        self.data_instances.append(sInstance)

        self.dataset_size = len(self.data_instances)
        print("\n{:s}\t{:s}\t{:d}".format(self.dataset_name, self.split, self.dataset_size))

    def __getitem__(self, index):

        s_instance = self.data_instances[index]


        s_feature = torch.from_numpy(self.features[s_instance.video_name][s_instance.position_vector]).float()
        s_index = torch.LongTensor([s_instance.startId, s_instance.endId])
        if self.output_rdIdx:
            s_score = torch.FloatTensor([float(s_instance.gt_score)])
            rd_idx = torch.LongTensor([s_instance.rd_startId, s_instance.rd_endId])
            rd_score = torch.FloatTensor([float(s_instance.rd_score)])
            return s_feature, s_index, s_score, rd_idx, rd_score

        if self.output_score:
            s_score = torch.FloatTensor([float(s_instance.gt_score)])

            return s_feature, s_index, s_score
        return s_feature, s_index

    def __len__(self):
        return self.dataset_size  # Predefine size...

if __name__ == '__main__':

    sDataset = Dataset(dataset_name='SumMe', split='train', sample_rates=[1, 5, 10], output_rdIdx=True)
    for i, (image, label) in enumerate(sDataset):

        print "DEBUG"