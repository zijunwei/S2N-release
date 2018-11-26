import torch
from PyUtils.AverageMeter import AverageMeter
import progressbar
from torch.autograd import Variable
import network
import torch.nn.functional as F
import numpy as np
import pickle
from operator import itemgetter

#TODO: check out https://gist.github.com/cabaf/30f7104743ff49af4cfc91e993a2c03b
class Evaluator():
    def __init__(self, dataloader, save_directory, savename='results.txt', unit_size=16.):
        self.dataloader = dataloader
        self.save_directory = save_directory
        self.savename = savename
        self.unit_size = unit_size
        self.movie_names = self.dataloader.dataset.movie_names
        self.len_prob = self.compute_prob_dist()



    def compute_prob_dist(self):
        clip_length_file = "/home/zwei/Dev/TURN_TAP_ICCV17/turn_codes/val_training_samples.txt"

        length_dist = {}
        for _key in [16, 32, 64, 128, 256, 512]:
            length_dist[_key] = 0
        with open(clip_length_file) as f:
            for line in f:
                clip_length = int(line.split(" ")[2]) - int(line.split(" ")[1])
                length_dist[clip_length] += 1
        sample_sum = sum([length_dist[_key] for _key in length_dist])
        prob = [float(length_dist[_key]) / sample_sum for _key in [16, 32, 64, 128, 256, 512]]
        return prob


    def posterior_probability_full(self, result_dict):
        for _key in result_dict:
            result_dict[_key] = sorted(result_dict[_key], key=itemgetter(2))[::-1]
            result_dict[_key] = np.array(result_dict[_key])
            x1 = result_dict[_key][:, 0]
            x2 = result_dict[_key][:, 1]
            s = result_dict[_key][:, 2]
            for k in range(x1.shape[0]):
                clip_length_index = [16, 32, 64, 128, 256, 512].index(
                    min([16, 32, 64, 128, 256, 512], key=lambda x: abs(x - int(x2[k] * 30 - x1[k] * 30))))
                #        print clip_length_index
                s[k] = s[k] * self.len_prob[clip_length_index]
            new_ind = np.argsort(s)[::-1]
            result_dict[_key] = result_dict[_key][new_ind, :]
        return result_dict

    def posterior_probability_class(self, result_dict):
        pass


    def evaluate(self, model, use_cuda, normalize=0):
        model.eval()

        result_dict = {}
        pbar = progressbar.ProgressBar(max_value=len(self.dataloader))
        for i_batch, sample_batched in enumerate(self.dataloader):
            pbar.update(i_batch)

            feature_batch = Variable(sample_batched[0])
            clip_batch = (sample_batched[1]).numpy()
            movie_name_index = sample_batched[2].numpy()

            if use_cuda:
                feature_batch = feature_batch.cuda()

            if normalize > 0:
                feature_batch = F.normalize(feature_batch, p=2, dim=1)

            output_v = model(feature_batch)
            cls_logits, loc_logits, loc_left_logits, loc_right_logits = network.extract_outputs(output_v)
            cls_probs = F.softmax(cls_logits, dim=1)

            reg_start = clip_batch[:,0] + loc_left_logits.data.cpu().numpy() * self.unit_size
            reg_end = clip_batch[:,1] + loc_right_logits.data.cpu().numpy() * self.unit_size
            round_reg_start = clip_batch[:,0] + np.round(loc_left_logits.data.cpu().numpy()) * self.unit_size
            round_reg_end = clip_batch[:,1] + np.round(loc_right_logits.data.cpu().numpy()) * self.unit_size
            action_score = cls_probs.data.cpu().numpy()[:, 1]
            output_probs = cls_probs.data.cpu().numpy()
            for s_idx in range(len(movie_name_index)):
                s_movie_name = self.movie_names[movie_name_index[s_idx][0]]
                s_reg_start = reg_start[s_idx]
                s_reg_end = reg_end[s_idx]
                s_round_reg_start = round_reg_start[s_idx]
                s_round_reg_end = round_reg_end[s_idx]
                s_action_score = action_score[s_idx]
                s_output = output_probs[s_idx]

                if s_movie_name not in result_dict:
                    result_dict[s_movie_name] = [[s_reg_start, s_reg_end, s_action_score]]
                    # result_dict_sliding[movie_name] = [[sliding_start, sliding_end, s_action_score]]
                else:
                    result_dict[s_movie_name].append([s_reg_start, s_reg_end, s_action_score])
                    # result_dict_sliding[movie_name].append([sliding_start, sliding_end, conf])
                # results_lst.append((s_movie_name, s_round_reg_start, s_round_reg_end, s_reg_start, s_reg_end, s_action_score,
                #                     s_output[0], s_output[1]))
        result_dict = self.posterior_probability_full(result_dict)

        # pickle.dump(results_lst, open("./{:s}/{:s}.pkl".format(self.save_directory, self.savename), "w"))


    def evaluate_raw(self, model, use_cuda, normalize=0):
        model.eval()
        results_lst = []

        pbar = progressbar.ProgressBar(max_value=len(self.dataloader))
        for i_batch, sample_batched in enumerate(self.dataloader):
            pbar.update(i_batch)

            feature_batch = Variable(sample_batched[0])
            clip_batch = (sample_batched[1]).numpy()
            movie_name_index = sample_batched[2].numpy()

            if use_cuda:
                feature_batch = feature_batch.cuda()

            if normalize > 0:
                feature_batch = F.normalize(feature_batch, p=2, dim=1)

            output_v = model(feature_batch)
            cls_logits, loc_logits, loc_left_logits, loc_right_logits = network.extract_outputs(output_v)
            cls_probs = F.softmax(cls_logits, dim=1)

            reg_start = clip_batch[:,0] + loc_left_logits.data.cpu().numpy() * self.unit_size
            reg_end = clip_batch[:,1] + loc_right_logits.data.cpu().numpy() * self.unit_size
            round_reg_start = clip_batch[:,0] + np.round(loc_left_logits.data.cpu().numpy()) * self.unit_size
            round_reg_end = clip_batch[:,1] + np.round(loc_right_logits.data.cpu().numpy()) * self.unit_size
            action_score = cls_probs.data.cpu().numpy()[:, 1]
            output_probs = cls_probs.data.cpu().numpy()
            for s_idx in range(len(movie_name_index)):
                s_movie_name = self.movie_names[movie_name_index[s_idx][0]]
                s_reg_start = reg_start[s_idx]
                s_reg_end = reg_end[s_idx]
                s_round_reg_start = round_reg_start[s_idx]
                s_round_reg_end = round_reg_end[s_idx]
                s_action_score = action_score[s_idx]
                s_output = output_probs[s_idx]
                results_lst.append((s_movie_name, s_round_reg_start, s_round_reg_end, s_reg_start, s_reg_end, s_action_score,
                                    s_output[0], s_output[1]))

        pickle.dump(results_lst, open("./{:s}/{:s}.pkl".format(self.save_directory, self.savename), "w"))