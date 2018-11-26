import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from PtUtils.DebugUtils import set_readable_param_names
class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, modulename=None):
        super(Attention, self).__init__()

        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.v = nn.Parameter(torch.FloatTensor(dim))

        if modulename is not None:
            self.project_query.weight.readable_name = '{:s}-Query-w'.format(modulename)
            self.project_query.bias.readable_name =  '{:s}-Query-b'.format(modulename)
            self.project_ref.weight.readable_name = '{:s}-Ref-w'.format(modulename)
            self.project_ref.bias.readable_name = '{:s}-Ref-b'.format(modulename)
            self.v.readable_name = '{:s}-Vec'.format(modulename)


        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))
        self.tanh = nn.Tanh()

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        logits = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        return e, logits