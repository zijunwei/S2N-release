import sys
import datasets.SumMe.path_vars as dataset_pathvars
import torch.utils.data as data
import numpy as np
import torch
videofile_stems = dataset_pathvars.file_names
videofile_stems.sort()

train_video_stems = videofile_stems[:20]
val_video_stems = videofile_stems[20:]
import vsSummDevs.JM.JKyLoader as KyLoader


def convertlabels2segs(labels):
    # print"NotImplemented"
    n_users = labels.shape[1]
    label_indicators = []
    for user_idx in range(n_users):
        s_labels = labels[:, user_idx]
        startIdx = 0
        endIdx = len(s_labels)
        for idx in range(1, len(s_labels)):
            if s_labels[idx] == 1 and s_labels[idx - 1] == 0:
                startIdx = idx
            if s_labels[idx] == 0 and s_labels[idx - 1] == 1:
                endIdx = idx
                if [startIdx, endIdx] not in label_indicators:
                    label_indicators.append([startIdx, endIdx])

    return label_indicators

def convertlabels2NonoverlappedSegs(labels):
    segments = convertlabels2segs(labels)
    boundaris = []
    for s_segment in segments:
        boundaris.extend(s_segment)
    boundaris = list(set(boundaris))
    boundaris = np.asarray(boundaris)
    idxes = np.argsort(boundaris)
    boundaris = boundaris[idxes]
    new_segments = []
    for i in range(1, len(boundaris)):
        new_seg = [boundaris[i-1], boundaris[i]]
        new_segments.append(new_seg)
    return new_segments


class Segment():
    def __init__(self, video_name):
        self.video_stem = video_name
    def initVector(self, indicatorVector):
        self.indicatorVector = indicatorVector
        indices = np.argwhere(indicatorVector==1)[0]
        self.startIdx = indices[0]
        self.endIdx = indices[-1]
        self.length = len(indicatorVector)
    def initId(self, startId, endId, length):
        self.indicatorVector = np.zeros(length)
        self.indicatorVector[startId:endId]=1
        self.startIdx = startId
        self.endIdx = endId
        self.length = length



def overlap(pred, true):
    n_elements = true.sum()
    n_overlap = (pred*true).sum()
    return n_overlap*1./n_elements

feature_set = {'ImageNet': [0, 1], 'Kinetics':[1, 2], 'Places':[2, 3], 'Moments': [3, 4], 'All':[0, 4]}


class Dataset(data.Dataset):

    def __init__(self, split='train', dataset_name='SumMe', clip_size=50):
        self.split = split
        dataset_path = '/home/zwei/datasets/KY_AAAI18/datasets/eccv16_dataset_{:s}_google_pool5.h5'.format(dataset_name.lower())
        self.dataset_name = dataset_name
        dataset = KyLoader.loadKyDataset(self.dataset_name, dataset_path)
        dataset_keys = KyLoader.getKyDatasetKeys(dataset)
        n_videos = len(dataset_keys)
        if self.split == 'train':
            self.video_stems = dataset_keys[1: int(n_videos*.8)]

        elif self.split == 'val':
            self.video_stems = dataset_keys[int(n_videos*.8):]

        else:
            print "Unrecognized data split: {:s}".format(split)
            sys.exit(-1)
        self.clip_size = clip_size
        self.videofeatures = {}
        self.annotations = []
        for s_key in self.video_stems:

            s_labels = dataset[s_key]['user_summary'][...]
            # n_frames = dataset[s_key]['n_frames'][()]
            s_video_features = dataset[s_key]['features'][...]
            positions = dataset[s_key]['picks'][...]

            s_labels = s_labels[:, positions]
            s_labels = s_labels.transpose()
            n_frames = s_video_features.shape[0]
            if s_video_features.shape[0] < clip_size:
                print "{:s} don't have enough frames, skipping".format(s_key)
                continue

            self.videofeatures[s_key] = s_video_features

            s_segs = convertlabels2segs(s_labels)
            for s_seg in s_segs:
                s_annotation = Segment(s_key)
                s_annotation.initId(startId=s_seg[0], endId=s_seg[1], length=n_frames)
                self.annotations.append(s_annotation)
        self.r_overlap = 0.3

        self.dataset_size = len(self.annotations)


    def __getitem__(self, index):

        s_annotation = self.annotations[index]

        frame_start = max(s_annotation.startIdx -25, 0)
        frame_end = frame_start + self.clip_size
        if frame_end > s_annotation.length:
            offset = frame_end-s_annotation.length
            frame_start -=offset
            frame_end = s_annotation.length


        s_groundtruth = torch.LongTensor([s_annotation.startIdx-frame_start, min(s_annotation.endIdx-frame_start, self.clip_size-1)])
        s_feature = torch.from_numpy(self.videofeatures[s_annotation.video_stem][frame_start:frame_end])




        return s_feature, s_groundtruth

    def __len__(self):
        return self.dataset_size # Predefine size...

if __name__ == '__main__':

    sDataset = Dataset(split='train')
    # HicoLoader = torch.utils.data.DataLoader(HicoDataset, batch_size=20, shuffle=False)
    for i, (image, label) in enumerate(sDataset):

        print "DEBUG"

