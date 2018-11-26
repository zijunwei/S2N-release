import torch
import torch.nn as nn
from torch.autograd import Variable

#TODO: bi-directional, multi-layer

def set_readable_param_names(model):
    for name, _ in model.named_parameters():
        getattr(model, name).readable_name = name



class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim, dropout=0.5, useGRU=False, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        #FIXME: now there is only 1 layer (double direction, dropout will be 1)
        if useGRU:
            self.rnn=nn.GRU(input_dim, hidden_dim, num_layers=num_layers,dropout=dropout, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)

        set_readable_param_names(self.rnn)
        # self.lstm.weight_hh_l0.readable_name = 'Encoder-weight_hh_l0'
        # self.lstm.weight_ih_l0.readable_name = 'Encoder-weight_ih_l0'
        # self.lstm.bias_hh_l0.readable_name = 'Encoder-bias_hh_l0'
        # self.lstm.bias_ih_l0.readable_name = 'Encoder-bias_ih_l0'
        # self.lstm.weight_hh_l0_reverse.readable_name = 'weight_hh_l0_reverse'

        # print("Encoder Created")
    def forward(self, x, hidden):
        # x [sourceL, batch_size, feature_dim]
        output, hidden = self.rnn(x, hidden)
        return output, hidden



def init_zero_hidden(hidden_dim, use_cuda=False):
        """Non-Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if use_cuda:
            enc_init_hx = enc_init_hx.cuda()

        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)
        if use_cuda:
            enc_init_cx = enc_init_cx.cuda()
        return (enc_init_hx, enc_init_cx)