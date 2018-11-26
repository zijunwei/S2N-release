#!/usr/bin/env python
# debug version 1:
# Evaluator


import argparse
import os,sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import pprint as pp
import numpy as np
import torch.nn.utils.clip_grad
import torch
print(torch.__version__)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PointerGRU2Heads_v4_3in import PointerNetwork
from PtUtils import cuda_model
from ActionLocalizationDevs.datasets.THUMOS14.single_video_evaluator_mscale_loader import SingleVideoLoader
import progressbar
import helper
import ActionLocalizationDevs.PropEval.Utils as PropUtils
import pandas as pd
def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser(description="Evaluation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--seq_len', default=90, type=int, help='clip size')
parser.add_argument('--net_outputs', default=15, type=int, help='number of intervals for lstm outputs')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')

# GPU
parser.add_argument("--gpu_id", default='1', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_dim', type=int, default=1024, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=1024, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
parser.add_argument('--eval', type=str, default='/home/zwei/Dev/NetModules/PtrNet2/gru2heads_proposal_s4-3_EMD_AUG_ckpt', help='check-point file')
parser.add_argument('--fileid', type=int, default=2, help='file id in ckpt file')

user_home_directory = os.path.expanduser('~')

def main():
    global args
    args = (parser.parse_args())
    ckpt_idx = args.fileid

    proposal_save_file = 'Dev/NetModules/ActionLocalizationDevs/PropEval/baselines_results/inception-s4-EMD-gru-aug-{:04d}_thumos14_test.csv'.format(ckpt_idx)
    feature_directory = os.path.join(user_home_directory, 'datasets/THUMOS14/features/BNInception')

    ground_truth_file = os.path.join(user_home_directory, '/home/zwei/Dev/NetModules/ActionLocalizationDevs/action_det_prep/thumos14_tag_test_proposal_list.csv')
    ground_truth = pd.read_csv(ground_truth_file, sep=' ')
    target_video_frms = ground_truth[['video-name', 'video-frames']].drop_duplicates().values
    frm_nums = {}
    for s_target_videofrms in target_video_frms:
        frm_nums[s_target_videofrms[0]] = s_target_videofrms[1]

    target_file_names = ground_truth['video-name'].unique()
    feature_file_ext = 'npy'

    use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)

    # Pretty print the run args
    pp.pprint(vars(args))

    model = PointerNetwork(input_dim=args.input_dim, embedding_dim=args.embedding_dim,
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs)


    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

    model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)
    model.eval()
    if args.eval is not None:
        # if os.path.isfile(args.resume):
        ckpt_filename = os.path.join(args.eval, 'checkpoint_{:04d}.pth.tar'.format(ckpt_idx))
        assert os.path.isfile(ckpt_filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(ckpt_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        train_iou = checkpoint['IoU']
        print("=> loading checkpoint '{}', current iou: {:.04f}".format(ckpt_filename, train_iou))


    predict_results = {}
    overlap=0.6
    seq_length = 90
    sample_rate = [1, 2, 4]
    for s_sample_rate in sample_rate:
        for video_idx, s_target_filename in enumerate(target_file_names):
            if not os.path.exists(os.path.join(feature_directory, '{:s}.{:s}'.format(s_target_filename, feature_file_ext))):
                print ('{:s} Not found'.format(s_target_filename))
                continue

            s_feature_path = os.path.join(feature_directory, '{:s}.{:s}'.format(s_target_filename, feature_file_ext))
            singlevideo_data = SingleVideoLoader(feature_path=s_feature_path, seq_length=seq_length, overlap=overlap, sample_rate=[s_sample_rate])
            n_video_len = singlevideo_data.n_features
            n_video_clips = len(singlevideo_data.video_clips)
            singlevideo_dataset = DataLoader(singlevideo_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

            predict_proposals = []

            for batch_idx, data in enumerate(singlevideo_dataset):
                    clip_feature = Variable(data[0], requires_grad=False)
                    clip_start_positions = Variable(data[1], requires_grad=False)
                    clip_end_positions = Variable(data[2], requires_grad=False)

                    if use_cuda:
                        clip_feature = clip_feature.cuda()
                        clip_start_positions = clip_start_positions.cuda()
                        clip_end_positions = clip_end_positions.cuda()

                    clip_start_positions = clip_start_positions.repeat(1, args.net_outputs)
                    clip_end_positions = clip_end_positions.repeat(1, args.net_outputs)

                    head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, _ = model(clip_feature)

                    cls_scores = F.softmax(cls_scores, dim=2)

                    head_positions, tail_positions = helper.reorder(head_positions, tail_positions)
                    head_positions = (head_positions*s_sample_rate + clip_start_positions)
                    tail_positions = (tail_positions*s_sample_rate + clip_start_positions)

                    cls_scores = cls_scores[:,:,1].contiguous().view(-1)
                    head_positions = head_positions.contiguous().view(-1)
                    tail_positions = tail_positions.contiguous().view(-1)

                    outputs = torch.stack([head_positions.float(), tail_positions.float(), cls_scores], dim=-1)
                    outputs = outputs.data.cpu().numpy()

                    for output_idx, s_output in enumerate(outputs):
                        if s_output[0] == s_output[1]:
                            s_output[0] -= s_sample_rate/2
                            s_output[1] += s_sample_rate/2
                            s_output[0] = max(0, s_output[0])
                            s_output[1] = min(n_video_len, s_output[1])
                            outputs[output_idx] = s_output

                    predict_proposals.append(outputs)

            predict_proposals = np.concatenate(predict_proposals, axis=0)
            predict_proposals, _ = PropUtils.non_maxima_supression(predict_proposals, overlap=0.999)
            # sorted_idx = np.argsort(predict_proposals[:,-1])[::-1]
            # predict_proposals = predict_proposals[sorted_idx]
            if s_target_filename in predict_results.keys():
                predict_results[s_target_filename] = np.concatenate((predict_results[s_target_filename], predict_proposals), axis=0)
            else:
                predict_results[s_target_filename] = predict_proposals

            n_proposals = len(predict_proposals)

            print("[{:d} | {:d}]{:s}\t {:d} Frames\t {:d} Clips\t{:d} Proposals @ rate:{:d}".
                  format(video_idx, len(target_file_names), s_target_filename, n_video_len, n_video_clips, n_proposals, s_sample_rate))

    data_frame = pkl_frame2dataframe(predict_results, frm_nums)
    results = pd.DataFrame(data_frame, columns=['f-end', 'f-init', 'score', 'video-frames', 'video-name'])
    results.to_csv(os.path.join(user_home_directory, proposal_save_file),
                   sep=' ', index=False)


def pkl_frame2dataframe(dt_results, frm_nums):
    data_frame = []
    print("Saving to cvs files.")
    pbar = progressbar.ProgressBar(max_value=len(dt_results))
    for i, _key in enumerate(dt_results):
        pbar.update(i)
        # fps = movie_fps[_key]
        frm_num = frm_nums[_key]
        for line in dt_results[_key]:
            start = int(line[0])
            end = int(line[1])
            if start == end:
                continue
            score = float(line[2])
            data_frame.append([end, start, score, frm_num, _key])
    return data_frame


if __name__ == '__main__':
    main()
