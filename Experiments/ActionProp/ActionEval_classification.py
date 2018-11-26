#!/usr/bin/env python
# debug version 1:
# Evaluator


import argparse
import os,sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/S2N-release')
sys.path.append(project_root)

import pprint as pp
import numpy as np
import torch.nn.utils.clip_grad
import torch
print(torch.__version__)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from SDN.PointerGRU2Heads_v8 import PointerNetwork
from PtUtils import cuda_model
from Devs_ActionProp.datasets.THUMOS14.single_video_evaluator_loader import SingleVideoLoader
import progressbar
import SDN.helper
from Devs_ActionProp.PropEval.Utils import non_maxima_supression
import pandas as pd
def str2bool(v):
      return v.lower() in ('true', '1', 'y', 'yes')

parser = argparse.ArgumentParser(description="Evaluation of Pointer-Net-LSTM-2Heads")
# Data
parser.add_argument('--seq_len', default=360, type=int, help='clip size')
parser.add_argument('--net_outputs', default=15, type=int, help='number of intervals for lstm outputs')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')

# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# Network
parser.add_argument('--input_dim', type=int, default=500, help='Number of hidden units')
parser.add_argument('--embedding_dim', type=int, default=500, help='Number of embedding units')
parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
parser.add_argument('--eval', type=str, default='/home/zwei/Dev/NetModules/ckpts/THUMOS/ActionExp_v3_l2_loss-NoDepsIn3-assgin0.50-alpha0.1000-dim512-dropout0.5000-seqlen90-L2-HUG', help='check-point file')
parser.add_argument('--fileid', type=int, default=15, help='file id in ckpt file')

user_home_directory = os.path.expanduser('~')

def main():
    global args
    args = (parser.parse_args())
    ckpt_idx = args.fileid
    savefile_stem = os.path.basename(args.eval)
    proposal_save_file = 'Dev/S2N-release/Devs_ActionProp/PropEval/baselines_results/{:s}-{:04d}-check-02_thumos14_test.csv'.format(savefile_stem, ckpt_idx)
    feature_directory = os.path.join(user_home_directory, 'datasets/THUMOS14/features/c3dd-fc7-red500')

    ground_truth_file = os.path.join(user_home_directory, 'Dev/NetModules/Devs_ActionProp/action_det_prep/thumos14_tag_test_proposal_list_c3dd.csv')
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
                           hidden_dim=args.hidden_dim, max_decoding_len=args.net_outputs, output_classes=2)


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
    overlap=0.9
    seq_length = 360
    sample_rate = 4

    for video_idx, s_target_filename in enumerate(target_file_names):
        if not os.path.exists(os.path.join(feature_directory, '{:s}.{:s}'.format(s_target_filename, feature_file_ext))):
            print ('{:s} Not found'.format(s_target_filename))
            continue

        s_feature_path = os.path.join(feature_directory, '{:s}.{:s}'.format(s_target_filename, feature_file_ext))
        singlevideo_data = SingleVideoLoader(feature_path=s_feature_path, seq_length=seq_length, overlap=overlap, sample_rate=sample_rate)
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
                # cls_scores = F.sigmoid(cls_scores)
                cls_scores = F.softmax(cls_scores, dim=2)

                head_positions, tail_positions = SDN.helper.switch_positions(head_positions, tail_positions)
                head_positions = (head_positions*sample_rate + clip_start_positions)
                tail_positions = (tail_positions*sample_rate + clip_start_positions)

                # cls_scores = cls_scores.contiguous().view(-1)
                cls_scores = cls_scores[:,:,1].contiguous().view(-1)
                head_positions = head_positions.contiguous().view(-1)
                tail_positions = tail_positions.contiguous().view(-1)

                outputs = torch.stack([head_positions.float(), tail_positions.float(), cls_scores], dim=-1)
                outputs = outputs.data.cpu().numpy()

                for output_idx, s_output in enumerate(outputs):
                    if s_output[0] == s_output[1]:
                        s_output[0] -= sample_rate/2
                        s_output[1] += sample_rate/2
                        s_output[0] = max(0, s_output[0])
                        s_output[1] = min(n_video_len, s_output[1])
                        outputs[output_idx] = s_output

                predict_proposals.append(outputs)

        predict_proposals = np.concatenate(predict_proposals, axis=0)
        sorted_idx = np.argsort(predict_proposals[:,-1])[::-1]
        predict_proposals = predict_proposals[sorted_idx]
        n_proposals = len(predict_proposals)
        pred_positions = predict_proposals[:, :2]
        pred_scores = predict_proposals[:,-1]
        nms_positions, nms_scores = non_maxima_supression(pred_positions, pred_scores, overlap=0.99)
        nms_predictions = np.concatenate((nms_positions, np.expand_dims(nms_scores, -1)), axis=-1)
        predict_results[s_target_filename] = predict_proposals

        print("[{:d} | {:d}]{:s}\t {:d} Frames\t {:d} Clips\t{:d} Proposals, Non-repeat:{:d}".
              format(video_idx, len(target_file_names), s_target_filename, n_video_len, n_video_clips, n_proposals, nms_predictions.shape[0]))

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
