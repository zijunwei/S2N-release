import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from modules.Encoder_bi_mlayer import Encoder
from modules.Decoder2Heads_GRU_v7_2 import Decoder

#Update: comapre to v4-mlayer-3in, in decoding stage for starting index, this will also based on hidden state
#Update: GRU version
# the auxilary part is removed.
# instead of pure 0s
# with new decoder take 3 vectors as inputs
#Update: compared to v7_regression, we modified the data flow in SBU
#Update: compared to v8, made a stupid mistake, everything is the same expect names
class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq 
    model"""
    def __init__(self,
            input_dim,
            embedding_dim,
            hidden_dim,
            max_decoding_len,dropout=0.5, aux=False, n_enc_layers=2, output_classes=1):  # TODO: added this elements in case we need states from previous sequences
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
        self.n_enc_layers = n_enc_layers
        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,dropout=dropout,useGRU=True, num_layers=self.n_enc_layers)

        self.decoder = Decoder(
                embedding_dim,
                hidden_dim*2,
                max_length=max_decoding_len, outputclasses=output_classes)
        self.aux = aux
        if self.aux:
            self.EncodeScore = nn.Linear(hidden_dim*2, 1)

        dec_init_0 = torch.FloatTensor(hidden_dim*2)

        enc_init_hx = torch.FloatTensor(hidden_dim)
        enc_init_cx = torch.FloatTensor(hidden_dim)

        dec_init_0.uniform_(-(1. / math.sqrt(hidden_dim*2)),
                1. / math.sqrt(hidden_dim*2))

        enc_init_hx.uniform_(-(1. / math.sqrt(hidden_dim)),
                1. / math.sqrt(hidden_dim))
        enc_init_cx.uniform_(-(1. / math.sqrt(hidden_dim)),
                1. / math.sqrt(hidden_dim))

        self.decoder_init_0 = nn.Parameter(dec_init_0, requires_grad=True)
        self.enc_init_cx = nn.Parameter(enc_init_cx, requires_grad=True)
        self.enc_init_hx = nn.Parameter(enc_init_hx, requires_grad=True)




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

        # (encoder_hx, encoder_cx) = init_zero_hidden(self.hidden_dim, inputs.is_cuda)
        # 2 or bi-directional LSTM
        #TODO: perhaps should split the initilization on different layers
        encoder_hx = self.enc_init_hx.unsqueeze(0).unsqueeze(0).repeat(2*self.n_enc_layers, batch_size, 1)
        # encoder_cx = self.enc_init_cx.unsqueeze(0).unsqueeze(0).repeat(2, batch_size, 1)
        
        # encoder forward pass
        enc_h, enc_h_t = self.encoder(embedded_inputs, encoder_hx)
        if self.aux:
            enc_h_linear = enc_h.view(-1, self.hidden_dim)
            enc_action_scores = self.EncodeScore(enc_h_linear)
            enc_action_scores = enc_action_scores.view(-1, batch_size).permute(1, 0)
        else:
            enc_action_scores = None

        dec_init_state = torch.cat([enc_h_t[-2], enc_h_t[-1]], dim=-1)

        # dec_init_state = (enc_h_t[-1], enc_c_t[-1])
    
        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_init_0.unsqueeze(0).repeat(batch_size, 1)

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
