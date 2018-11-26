import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from Attention2 import Attention
import math
#TODO: drived from Decoder2Heads_GRU_v3
class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 max_length):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.input_weights = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.input_weights.weight.readable_name = 'Decoder-ih-w'
        self.input_weights.bias.readable_name = 'Decoder-ih-b'

        self.hidden_weights = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.hidden_weights.weight.readable_name = 'Decoder-hh-w'
        self.hidden_weights.bias.readable_name = 'Decoder-hh-b'
        # self.input_reduction = nn.Linear(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, modulename='pointer')

        self.cls = torch.nn.Conv1d(2, 2, kernel_size=hidden_dim, stride=1, padding=0)
        self.cls.weight.readable_name = 'Decoder-ScoreConv-w'
        self.cls.bias.readable_name = 'Decoder-ScoreConv-b'
        self.sm = nn.Softmax(dim=1)


    def forward(self, decoder_input, hidden, context, max_len=None):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        batch_size = context.size(1)
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


            _, logits = self.pointer(hy, context)

            encs, idxs = self.decode_argmax(logits,context)

            #TODO: double check here
            feature = torch.stack([encs, hy], dim=1)

            cls_score = self.cls(feature).squeeze(2)

            return hy, logits, idxs, cls_score



        pointer_probs = []
        pointer_idxes = []
        cls_scores = []

        cur_max_len = max_len or self.max_length

        steps = range(cur_max_len)  # or until terminating symbol ?

        for _ in steps:
            hx, probs, idxs, cls_score  = recurrence(decoder_input, hidden)
            hidden = hx
            # select the next inputs for the decoder [batch_size x hidden_dim]
            #TODO: decoder input can also come from the training data
            # decoder_input, idxs_head = self.decode_argmax(
            #     probs_s,
            #     context) # change embedding inputs to context

            # _, idxs_tail = self.decode_argmax(probs_e)
            pointer_probs.append(probs)
            pointer_idxes.append(idxs)

            cls_scores.append(cls_score)

        return (pointer_probs, pointer_idxes, cls_scores), hidden


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
        _, idxs = probs.max(dim=1)


        if embedded_inputs is not None:
            batch_size = probs.size(0)
            sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]
        else:
            sels = None

        return sels, idxs


if __name__ == '__main__':
    decoder = Decoder(hidden_dim=36, max_length=10)
    decoder_input = Variable(torch.randn([10, 36]))
    hidden = Variable(torch.randn([10, 36]))
    context = Variable(torch.randn([50, 10, 36]))

    (pointer_probs, pointer_idxes, cls_scores), decoded_stats = decoder(decoder_input, hidden, context)
    print("DEBUG")