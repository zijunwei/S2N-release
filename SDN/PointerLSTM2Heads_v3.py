import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from modules.Encoder2 import Encoder, init_zero_hidden
from modules.Decoder2Heads_LSTM_v2 import Decoder


#TODO compared to V2, this will add some linear score predictor based on encoding stage features
class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""
    def __init__(self,
            input_dim,
            embedding_dim,
            hidden_dim,
            max_decoding_len,dropout=0):  # TODO: added this elements in case we need states from previous sequences
        super(PointerNetwork, self).__init__()
        # self.use_cuda = use_cuda

        if input_dim != embedding_dim:
            embedding_ = torch.FloatTensor(input_dim, embedding_dim)
            if self.use_cuda:
                embedding_ = embedding_.cuda()
            self.embedding = nn.Parameter(embedding_)
            self.embedding.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                         1. / math.sqrt(embedding_dim))
            self.embed_input = True
        else:
            self.embed_input = False


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,dropout=dropout)

        self.decoder = Decoder(
                embedding_dim,
                hidden_dim,
                max_length=max_decoding_len)
        self.EncodeScore = nn.Linear(hidden_dim, 1)

        dec_in_0 = torch.FloatTensor(hidden_dim)
        dec_in_0.uniform_(-(1. / math.sqrt(hidden_dim)),
                1. / math.sqrt(hidden_dim))

        self.decoder_in_0 = nn.Parameter(dec_in_0, requires_grad=True)
        # self.decoder_in_0.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #         1. / math.sqrt(hidden_dim))


    def forward(self, inputs, decode_len=None):
        """ Propagate inputs through the network
        Args:
            inputs: [batch_size, input_dim, sourceL]

        """

        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        assert input_dim == self.input_dim, 'input dim should be {:d} but now: {:d}'.format(self.input_dim, input_dim)

        sourceL = inputs.size(2)

        if self.embed_input:
            # repeat embeddings across batch_size
            # result is [batch_size x input_dim x embedding_dim]
            # TODO: repeat or expand?
            embedding = self.embedding.repeat(batch_size, 1, 1)
            embedded_inputs = []
            # result is [batch_size, 1, input_dim, sourceL]
            ips = inputs.unsqueeze(1)

            for i in range(sourceL):
                # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
                # result is [batch_size, embedding_dim]
                embedded_inputs.append(torch.bmm(
                    ips[:, :, :, i].float(),
                    embedding).squeeze(1))

            # Result is [sourceL x batch_size x embedding_dim]
            embedded_inputs = torch.cat(embedded_inputs).view(
                sourceL,
                batch_size,
                embedding.size(2))
        else:
            embedded_inputs = inputs.permute(2, 0, 1)

        (encoder_hx, encoder_cx) = init_zero_hidden(self.hidden_dim, inputs.is_cuda)
        encoder_hx = encoder_hx.unsqueeze(0).repeat(embedded_inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(embedded_inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(embedded_inputs, (encoder_hx, encoder_cx))

        enc_h_linear = enc_h.view(-1, self.hidden_dim)
        # enc_h_linear_2d = enc_h_linear.view(self.hidden_dim, -1)
        enc_action_scores = self.EncodeScore(enc_h_linear)
        enc_action_scores = enc_action_scores.view(-1, batch_size).permute(1, 0)
        dec_init_state = (enc_h_t[-1], enc_c_t[-1])
    
        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(embedded_inputs.size(1), 1)

        (head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores), dec_hidden_t = self.decoder(decoder_input,
                embedded_inputs,
                dec_init_state,
                enc_h, max_len=decode_len)
        #TODO: added conversion to tensors
        head_pointer_probs = torch.stack(head_pointer_probs)
        head_pointer_probs = head_pointer_probs.permute(1, 0, 2)
        tail_pointer_probs = torch.stack(tail_pointer_probs)
        tail_pointer_probs = tail_pointer_probs.permute(1, 0, 2)
        cls_scores = torch.stack(cls_scores)
        cls_scores = cls_scores.permute(1, 0, 2)
        head_positions = torch.stack(head_positions)
        head_positions = head_positions.permute(1, 0)
        tail_positions = torch.stack(tail_positions)
        tail_positions = tail_positions.permute(1, 0)



        return head_pointer_probs, head_positions, tail_pointer_probs, tail_positions, cls_scores, enc_action_scores
