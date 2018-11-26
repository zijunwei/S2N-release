import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding by repeating!)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        input = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))
        # input = input.unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, F.tanh(input + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = F.softmax(att, dim=1)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, att

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, state_dim,
                 hidden_dim, output_dim=None):

        super(Decoder, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # here the LSTM is hand-created...
        self.hidden_out = nn.Linear(self.hidden_dim * 3, output_dim)
        self.attention_start = Attention(hidden_dim, hidden_dim)
        self.attention_end = Attention(hidden_dim, hidden_dim)

        self.mask_helper = Parameter(torch.zeros(1), requires_grad=False)
        self.index_helper = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, decoder_input,
                context):

        # hidden = init_hidden
        batch_size = context.size(0)
        input_length = context.size(1)

        masks = self.mask_helper.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)

        self.attention_start.init_inf(masks.size())
        self.attention_end.init_inf(masks.size())

        runner = self.index_helper.repeat(input_length)
        #TODO: how about using cumsum?
        # runner = torch.cumsum(runner, dim=0) - 1
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()


        start_vector, starting_idx_vector = self.attention_start(decoder_input, context, torch.eq(masks, 1))
        start_probabilities, start_indices = starting_idx_vector.max(dim=1)

        # masks = (runner <= start_indices.unsqueeze(1).expand(-1, starting_idx_vector.size()[1])).float()
        onehot_start_masks = (runner == start_indices.unsqueeze(1).expand(-1, starting_idx_vector.size()[1])).float()

        end_vector, ending_idx_vector = self.attention_end(decoder_input, context, torch.eq(masks, 1))
        # ending_idx_vector = ending_idx_vector * (1-masks)
        end_probabilitis, end_indices = ending_idx_vector.max(dim=1)
        # end_masks = (runner <= end_indices.unsqueeze(1).expand(-1, ending_idx_vector.size()[1])).float()
        onehot_end_masks = (runner == end_indices.unsqueeze(1).expand(-1, ending_idx_vector.size()[1])).float()

        context_start_mask = onehot_start_masks.unsqueeze(2).expand(-1, -1, self.hidden_dim).byte()
        context_end_mask = onehot_end_masks.unsqueeze(2).expand(-1, -1, self.hidden_dim).byte()
        # context_g_masks = (runner == (input_length-1)).unsqueeze(-1).expand(-1, -1, self.hidden_dim).byte()


        score_feature_vector = torch.cat([context[context_start_mask].view(batch_size, self.hidden_dim),
                                 context[context_end_mask].view(batch_size, self.hidden_dim),
                                 decoder_input], 1)
        output_score = self.hidden_out(score_feature_vector)

        index_vector = torch.cat([starting_idx_vector.unsqueeze(-1), ending_idx_vector.unsqueeze(-1)], dim=-1).permute(0, 2, 1)
        return index_vector, output_score



class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout=0,
                 encoder_bidir=True, cls_output_dim=1):

        super(PointerNet, self).__init__()
        self.input_dim = input_dim
        self.encoder_bidir = encoder_bidir
        self.encoder = Encoder(input_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               encoder_bidir)
        self.cls_output = cls_output_dim
        self.decoder = Decoder(hidden_dim, hidden_dim, output_dim=self.cls_output)

    def forward(self, inputs):

        # batch_size = inputs.size(0)
        # input_length = inputs.size(1)

        encoder_hidden0 = self.encoder.init_hidden(inputs)
        encoder_outputs, encoder_hidden = self.encoder(inputs,
                                                       encoder_hidden0)
        if self.encoder_bidir:
            decoder_input = torch.cat([encoder_hidden[0][-2], encoder_hidden[0][-1]], dim=-1)  # This is concatnating **h_n**'s last layers backward and forward
        else:
            decoder_input = torch.cat([encoder_hidden[0][-1],
                               encoder_hidden[1][-1]], dim=-1)

        index_vector, values = self.decoder(decoder_input,
                                                           encoder_outputs)

        return  index_vector, values