import torch
import torch.nn as nn
from torch.autograd import Variable

#TODO: bi-directional, multi-layer
class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim,dropout=dropout)
        #TODO: comment later
        self.lstm.weight_hh_l0.readable_name = 'Encoder-weight_hh_l0'
        self.lstm.weight_ih_l0.readable_name = 'Encoder-weight_ih_l0'
        self.lstm.bias_hh_l0.readable_name = 'Encoder-bias_hh_l0'
        self.lstm.bias_ih_l0.readable_name = 'Encoder-bias_ih_l0'
        # print("Encoder Created")
    def forward(self, x, hidden):
        # x [sourceL, batch_size, feature_dim]
        output, hidden = self.lstm(x, hidden)
        return output, hidden



def init_zero_hidden(hidden_dim, use_cuda=False):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if use_cuda:
            enc_init_hx = enc_init_hx.cuda()

        # enc_init_hx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if use_cuda:
            enc_init_cx = enc_init_cx.cuda()

        # enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return (enc_init_hx, enc_init_cx)