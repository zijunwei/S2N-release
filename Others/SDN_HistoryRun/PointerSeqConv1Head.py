import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from modules.Encoder_conv import Encoder
from modules.Decoder1Heads_GRU import Decoder


# Update: GRU version
# the auxilary part is removed.
# instead of pure 0s
# with new decoder take 3 vectors as inputs
class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq
    model"""

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 max_decoding_len, dropout=0):
        super(PointerNetwork, self).__init__()


        if input_dim != embedding_dim:
            embedding_ = torch.FloatTensor(input_dim, embedding_dim).uniform_(-(1. / math.sqrt(embedding_dim)),
                                         1. / math.sqrt(embedding_dim))

            self.embedding = nn.Parameter(embedding_, requires_grad=True)

            self.embed_input = True
        else:
            self.embed_input = False

        self.input_dim = input_dim
        self.embeding_dim = embedding_dim


        self.encoder = Encoder(hidden_dim=embedding_dim, kernel_size=3, n_blocks=3, dropout=dropout)

        self.decoder = Decoder(hidden_dim=embedding_dim, max_length=max_decoding_len)


        dec_init_0 = torch.FloatTensor(embedding_dim).uniform_(-(1. / math.sqrt(embedding_dim)),
                            1. / math.sqrt(embedding_dim))

        self.decoder_init_0 = nn.Parameter(dec_init_0, requires_grad=True)


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
            embedded_inputs = embedded_inputs.permute(1, 2, 0)
        else:
            # embedded_inputs = inputs.permute(2, 0, 1)
            embedded_inputs = inputs

        # encoder forward pass
        enc_h  = self.encoder(embedded_inputs)

        dec_init_state = enc_h[:,:,-1]
        enc_h = enc_h.permute(2, 0, 1)

        # dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_init_0.unsqueeze(0).repeat(batch_size, 1)

        (pointer_probs, pointer_idxes,
         cls_scores), dec_hidden_t = self.decoder(decoder_input,
                                                  dec_init_state,
                                                  enc_h, max_len=decode_len)
        # TODO: added conversion to tensors
        pointer_probs = torch.stack(pointer_probs)
        pointer_probs = pointer_probs.permute(1, 0, 2)
        cls_scores = torch.stack(cls_scores)
        cls_scores = cls_scores.permute(1, 0, 2)
        pointer_idxes = torch.stack(pointer_idxes)
        pointer_idxes = pointer_idxes.permute(1, 0)


        return pointer_probs, pointer_idxes, cls_scores


if __name__ == '__main__':
    batch_size = 1
    input_dim = 256
    embedding_dim = 128
    input_len = 27
    import graph_vis

    x = Variable(torch.randn([batch_size, input_dim, input_len]))
    model = PointerNetwork(input_dim, embedding_dim, max_decoding_len=3)
    print("Number of Params\t{:d}".format(sum([p.data.nelement() for p in model.parameters()])))

    pred_probs, pred_positions, pred_scores = model(x)
    pred_probs_node = torch.sum(pred_probs)
    pred_positions_node = torch.sum(pred_positions).float()
    pred_scores_node = torch.sum(pred_scores)

    dot = graph_vis.make_dot(pred_positions_node + pred_probs_node + pred_scores_node)
    dot.save('graph-onepointer.dot')
    print("DEBUG")