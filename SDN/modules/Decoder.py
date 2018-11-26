import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# from PtrNet2.utils.beam_search import Beam
from Attention import Attention

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_length,
                 tanh_exploration,
                 terminating_symbol, # what does this do?
                 use_tanh,
                 decode_type,
                 n_glimpses=1,
                 beam_size=0,
                 use_cuda=True):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol
        self.decode_type = decode_type
        self.beam_size = beam_size
        self.use_cuda = use_cuda

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, use_cuda=self.use_cuda)
        self.glimpse = Attention(hidden_dim, use_tanh=False, use_cuda=self.use_cuda)
        self.sm = nn.Softmax(dim=1)

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):
        if mask is None:
            mask = torch.zeros(logits.size()).type_as(logits).byte()
            # if self.use_cuda:
            #     mask = mask.cuda()
        #TODO: Check here!
        maskk = mask.clone()

        # to prevent them from being reselected.
        # Or, allow re-selection and penalize in the objective function
        if prev_idxs is not None:
            # set most recently selected idx values to 1
            maskk[[x for x in range(logits.size(0))],
                  prev_idxs.data] = 1
            logits[maskk] = -np.inf
        return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        def recurrence(x, hidden, logit_mask, prev_idxs, step):

            hx, cx = hidden  # batch_size x hidden_dim

            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # batch_size x hidden_dim

            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
                # [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(self.max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None

        if self.decode_type == "stochastic":
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs = self.decode_stochastic(
                    probs,
                    embedded_inputs,
                    selections)
                inps.append(decoder_input)
                # use outs to point to next object
                outputs.append(probs)
                selections.append(idxs)
            return (outputs, selections), hidden

        elif self.decode_type == 'argmax':
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs = self.decode_argmax(
                    probs,
                    embedded_inputs,
                    selections)
                inps.append(decoder_input)
                # use outs to point to next object
                outputs.append(probs)
                selections.append(idxs)
            return (outputs, selections), hidden
        else:
            print "decoding type {:s} not recognized".format(self.decode_type)

    def decode_stochastic(self, probs, embedded_inputs, selections):
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
        batch_size = probs.size(0)
        # idxs is [batch_size]
        idxs = probs.multinomial().squeeze(1)

        # due to race conditions, might need to resample here
        for old_idxs in selections:
            # compare new idxs
            # elementwise with the previous idxs. If any matches,
            # then need to resample
            if old_idxs.eq(idxs).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
                break

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]
        return sels, idxs

    def decode_argmax(self, probs, embedded_inputs, selections):
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
        batch_size = probs.size(0)
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

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]
        return sels, idxs


