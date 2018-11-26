import torch
from Encoder import Encoder
from torch import nn

class BiLSTMClsLocalizer(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir, output_dim=3):
        super(BiLSTMClsLocalizer, self).__init__()

        self.encoder = Encoder(input_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir)

        self.linear = nn.Linear(hidden_dim, output_dim)


    def forward(self, input):

        outputs, encoder_states = self.encoder(input)
        outputs = self.linear(outputs)
        return outputs


class BiLSTMEMDLocalizer(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        super(BiLSTMClsLocalizer, self).__init__()

        self.encoder = Encoder(input_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir)

        self.start_linear = nn.Linear(hidden_dim, 1)
        self.end_linear = nn.Linear(hidden_dim, 1)

    def forward(self, input):

        outputs, encoder_states = self.encoder(input)
        start_idxes = self.start_linear(outputs)
        end_idxes = self.end_linear(outputs)

        return torch.cat([start_idxes, end_idxes], dim=-1)