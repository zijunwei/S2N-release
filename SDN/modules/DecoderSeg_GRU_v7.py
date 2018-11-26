import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from Attention_v2 import Attention
from PtUtils.DebugUtils import set_readable_param_names

import math
#TODO: compared to v1, motivated according to week 13
#TODO: check the gradient flow
#Update: GRU version
#Update: compared to v2, the cls is also based on the current hidden state of decoder
#Update: compared to v3, the starting index is based on the hidden state
#Update: compared to v4, the decoder input is based on previous decodings
#Update: compared to v5, the score output will be a regression!
#Update: compared to v6, we have updated the flow process!
#Update: compared to v7, just decoding 1 pointer and 1score
class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_length, outputclasses=1):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.input_weights = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.input_weights.weight.readable_name = 'Decoder-ih-w'
        self.input_weights.bias.readable_name = 'Decoder-ih-b'

        self.hidden_weights = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.hidden_weights.weight.readable_name = 'Decoder-hh-w'
        self.hidden_weights.bias.readable_name = 'Decoder-hh-b'
        # self.input_reduction = nn.Linear(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, modulename='Pointer')


        self.score_conv = torch.nn.Linear(hidden_dim, hidden_dim / 2)

        self.score_linear = torch.nn.Linear(hidden_dim / 2, outputclasses)

        self.sm = nn.Softmax(dim=1)




    def forward(self, decoder_input, embedded_inputs, hidden, context, max_len=None):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        batch_size = context.size(1)
        # prev_e_enc = self.prev_e_enc.unsqueeze(0).repeat(batch_size, 1)
        # prev_s_enc = self.prev_s_enc.unsqueeze(0).repeat(batch_size, 1)
        def recurrence(x, hidden):

            hx = hidden  # batch_size x hidden_dim

            gates1 = self.input_weights(x)
            gates2 = self.hidden_weights(hx)
            i_r, i_z, i_n = gates1.chunk(3, 1)
            h_r, h_z, h_n = gates2.chunk(3, 1)

            r_t = F.sigmoid(i_r + h_r)
            z_t = F.sigmoid(i_z + h_z)
            n_t = F.tanh(i_n + r_t*h_n)
            hy = (1 - z_t)*n_t + z_t * hx

            # ingate, forgetgate, cellgate = gates.chunk(3, 1)

            # ingate = F.sigmoid(ingate)
            # forgetgate = F.sigmoid(forgetgate)
            # cellgate = F.tanh(cellgate + ingate*)
            # outgate = F.sigmoid(outgate)
            # n_t =
            # cy = (forgetgate * cx) + (ingate * cellgate)
            # hy = outgate * F.tanh(cy)  # batch_size x hidden_dim

            g_l = hy




            _, logits = self.pointer(g_l, context)
            # logits_e, mask_tail = self.apply_mask_to_logits(step, logits_e, mask_tail, prev_idxs_tail)
            # probs_s = self.sm(logits_s)
            g_l_s, idxs = self.decode_argmax(logits,context)

            # feature = torch.stack([g_l_s, g_l_e, g_l], dim=-1)

            # feature = F.relu((g_l))

            score_intermediate = F.dropout(F.relu(self.score_conv(g_l)), 0.5)

            cls_score = self.score_linear(score_intermediate.view(score_intermediate.size(0), -1))


            next_input = None

            return hy, next_input, logits, idxs, cls_score

        pointer_probs = []
        outputs_tail = []

        pointer_positions = []
        selections_tail = []
        cls_scores = []
        # if max_len is not None:
        #     cur_max_len = max_len
        # else:
        #     cur_max_len = self.max_length

        cur_max_len = max_len or self.max_length

        steps = range(cur_max_len)  # or until terminating symbol ?
        # idxs_head = None
        # idxs_tail = None
        # mask_head = None
        # mask_tail = None

        # decoder_input = self.input_reduction(decoder_input)

        for i in steps:
            hx, _, probs, idxs, cls_score = \
                recurrence(decoder_input, hidden)
            hidden = hx

            # select the next inputs for the decoder [batch_size x hidden_dim]
            #TODO: decoder input can also come from the training data
            # decoder_input, idxs_head = self.decode_argmax(
            #     probs_s,
            #     context) # change embedding inputs to context

            # _, idxs_tail = self.decode_argmax(probs_e)

            pointer_probs.append(probs)

            pointer_positions.append(idxs)
            cls_scores.append(cls_score)

        return pointer_probs, pointer_positions, cls_scores, hidden


    def decode_argmax(self, probs, embedded_inputs=None):
        """
        Return the next input for the decoder by selecting the
        input corresponding to the max output

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indicies
        """
        # idxs is [batch_size]
        # idxs = probs.multinomial().squeeze(1)
        _, idxs = probs.max(dim=1)

        # due to race conditions, might need to resample here
        # for old_idxs in selections:
        #     # compare new idxs
        #     # elementwise with the previous idxs. If any matches,
        #     # then need to resample
        #     if old_idxs.eq(idxs).data.any():
        #         print(' [!] resampling due to race condition')
        #         idxs = probs.multinomial().squeeze(1)
        #         break

        if embedded_inputs is not None:
            batch_size = probs.size(0)
            sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]
        else:
            sels = None

        return sels, idxs


