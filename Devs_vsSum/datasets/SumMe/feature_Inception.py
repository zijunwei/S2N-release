import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/VideoSum')
sys.path.append(project_root)

import argparse
import SumCodes.Adversarial_Video_Summary.ImageDataset as ImageDataset
import SumCodes.Adversarial_Video_Summary.InceptionBackbone as InceptionBackbone
import glob
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.utils.data
from torch.autograd import Variable
import datasets.SumMe.path_vars as SumMe_pathvars
import scipy.io
import datasets.SumMe.LoadLabels as LoadLabels
import PyUtils.dir_utils as dir_utils
import progressbar

parser = argparse.ArgumentParser(description="Extract Visual Features from Frames")
parser.add_argument("--gpu_id", default=None, type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
parser.add_argument('--batch_size', '-bs', default=20, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='verbose mode, if not, saved in log.txt')
parser.add_argument('--network', default=None, type=str)
parser.add_argument('--pick', default=None, type=int)


#TODO: add labels.
args = parser.parse_args()
if args.network is None:
    args.network = 'InceptionV3'

frame_root = SumMe_pathvars.dst_frame_dir
frame_folder_list = glob.glob(os.path.join(frame_root, '*/'))

Inception = InceptionBackbone.inception_v3(pretrained=True)
Inception.eval()

useCuda = torch.cuda.is_available() and (args.gpu_id is not None or args.multiGpu)
if useCuda:
    if args.multiGpu:
        if args.gpu_id is None: # using all the GPUs
            device_count = torch.cuda.device_count()
            print("Using ALL {:d} GPUs".format(device_count))
            model = nn.DataParallel(Inception, device_ids=[i for i in range(device_count)]).cuda()
        else:
            print("Using GPUs: {:s}".format(args.gpu_id))
            device_ids = [int(x) for x in args.gpu_id]
            model = nn.DataParallel(Inception, device_ids=device_ids).cuda()


    else:
        torch.cuda.set_device(int(args.gpu_id))
        Inception.cuda()

    cudnn.benchmark = True
if args.pick is not None:
    save_dir = dir_utils.get_dir(os.path.join(SumMe_pathvars.dataset_dir, 'Annotation_{:s}_{:.2d}'.format(args.network, args.pick)))
else:
    save_dir = dir_utils.get_dir(os.path.join(SumMe_pathvars.dataset_dir, 'Annotation_{:s}'.format(args.network)))
for video_idx, s_frame_folder in enumerate(frame_folder_list):
    frame_feature = []
    s_video_stem = (s_frame_folder.split(os.sep))[-2]
    print "[{:02d} | {:02d}]Processing {:s}".format(video_idx, len(frame_folder_list), s_video_stem)

    label_file = os.path.join(SumMe_pathvars.ground_truth_dir, '{:s}.mat'.format(s_video_stem))
    gt_mat = LoadLabels.getRawData(label_file)
    labels = LoadLabels.getUserScores(gt_mat)
    if args.pick is not None:
        labels = labels[::args.pick,:]

    s_frame_list = glob.glob(os.path.join(s_frame_folder,'*.jpg'))
    s_frame_list.sort()
    if args.pick is not None:

        s_frame_list = ImageDataset.pick_EveryN(s_frame_list, args.pick)

    s_frame_dataset = ImageDataset.ImageDataset(s_frame_list)
    s_frame_loader = torch.utils.data.DataLoader(s_frame_dataset,
                                                        batch_size=args.batch_size, shuffle=False,
                                                        num_workers=args.workers, drop_last=False)
    pbar = progressbar.ProgressBar(max_value=len(s_frame_list)//args.batch_size)
    for i, s_image_batch in enumerate(s_frame_loader):
        pbar.update(i)
        if useCuda:
            s_image_batch = s_image_batch.cuda()

        v_image_batch = Variable(s_image_batch, volatile=True)

        res5c, pool5 = Inception(v_image_batch)
        pool5 = pool5.squeeze(2)
        pool5 = pool5.squeeze(2)
        local_feature = pool5.data.cpu().numpy().tolist()
        frame_feature.extend(local_feature)

    frame_feature = np.asarray(frame_feature)

    save_dict = {'features': frame_feature, 'labels': labels}
    save_file = os.path.join(save_dir, '{:s}.mat'.format(s_video_stem))
    scipy.io.savemat(save_file, save_dict)








