# Loading single video feature to be processed by the ...

import torch.utils.data as data
import torch
#Update: trained based on generated ground truth from https://github.com/yjxiong/temporal-segment-networks
import progressbar
import numpy as np




miss_name = ('video_test_0001292','video_validation_0000190')

class SingleVideoLoader(data.Dataset):

    def __init__(self, feature_path, seq_length=90, overlap=0.9, sample_rate=None):
        self.feature = np.load(feature_path)
        self.n_features = self.feature.shape[0]
        self.video_clips = []
        self.seq_len = seq_length
        self.sample_rate = sample_rate or [1, 2, 4]

        if self.n_features < seq_length:
            print("{:s} Do not have enough frames: {:d}".format(feature_path, self.n_features))
            return

        for s_sample_rate in self.sample_rate:

            start_idx = 0
            isInbound = True

            while start_idx < self.n_features and isInbound:
                end_idx = start_idx + seq_length*s_sample_rate
                if end_idx >= self.n_features:
                    isInbound = False
                    start_idx = start_idx - (end_idx-self.n_features)
                    end_idx = self.n_features
                self.video_clips.append((start_idx, end_idx, s_sample_rate))
                start_idx = int(start_idx + seq_length*(1-overlap)*s_sample_rate)

    def __getitem__(self, index):
        # ONLY in this part the sample rate jumps in
        s_instance = self.video_clips[index]
        # s_full_feature = np.load(os.path.join(self.feature_directory, '{:s}.{:s}'.format(s_instance['name'], self.feature_file_ext)))
        clip_start_position = s_instance[0]
        clip_end_position = s_instance[1]
        sample_rate = s_instance[2]
        s_clip_feature = self.feature[(clip_start_position):(clip_end_position):sample_rate]
        assert s_clip_feature.shape[0] == self.seq_len, 'feature size wrong!'

        s_clip_feature = torch.FloatTensor(s_clip_feature).transpose(0, 1)
        clip_start_position = torch.LongTensor([clip_start_position])
        clip_end_position = torch.LongTensor([clip_end_position])
        return s_clip_feature, clip_start_position, clip_end_position


    def __len__(self):

        return len(self.video_clips)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import glob
    import os
    feature_paths = glob.glob(os.path.join('/home/zwei/datasets/THUMOS14/features/BNInception', '*.npy'))

    for s_feature_path in feature_paths:
        thumos_dataset = SingleVideoLoader(s_feature_path,seq_length=360, overlap=0.9, sample_rate=4)
        train_dataloader = DataLoader(thumos_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=4)

        pbar = progressbar.ProgressBar(max_value=len(train_dataloader))
        offline_dataset = {}
        for idx, data in enumerate(train_dataloader):
            pbar.update(idx)
            print("DEVB")

