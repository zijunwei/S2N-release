import torch
import torch.nn as nn
import torch.nn.functional as F
import PtUtils.DebugUtils as utils




#TODO: add normalizations and good initilizations
class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, hidden_dim, kernel_size=3, n_blocks=2, dropout=0):
        super(Encoder, self).__init__()

        self.encoder = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=kernel_size)
        utils.set_readable_param_names(self.encoder, 'Enc-Conv-output')
        self.gate = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=kernel_size)
        utils.set_readable_param_names(self.gate, 'Enc-Conv-gate')
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.norm = nn.BatchNorm1d(num_features=hidden_dim)

    def forward(self, x):
        outputs = []
        outputs.append(x)
        for _ in range(self.n_blocks):
            conv_output = self.encoder(x)
            conv_gate = self.gate(x)
            x = F.sigmoid(conv_gate) * conv_output
            x = F.relu(x)

            # added normalization here:
            outputs.append(x)

        outputs = torch.cat(outputs, dim=2)
        outputs = self.norm(outputs)

        # Add normalization!
        if self.dropout>0:
            outputs = F.dropout(outputs, self.dropout)

        return outputs


if __name__ == '__main__':
    from torch.autograd import Variable
    # batch_size, input_dim, input_len
    x = Variable(torch.randn([1, 36, 27]))
    model = Encoder(hidden_dim=36, kernel_size=3, n_blocks=3)
    y = model(x)
    print("DEBUG")



