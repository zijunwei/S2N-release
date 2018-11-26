import torch
import torch.nn as nn
import torch.nn.functional as F
class TURN(nn.Module):
    def __init__(self, mid_layer_size=1000, feature_size=2048, output_size=4, drop=0.5):
        super(TURN, self).__init__()
        self.input_layer_size = feature_size * 3
        self.mid_layer_size = mid_layer_size
        self.linear1 = nn.Linear(self.input_layer_size, self.mid_layer_size)
        self.dropout = drop
        self.linear2 = nn.Linear(self.mid_layer_size, output_size)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.linear2(x)
        return x


def extract_outputs(output):
    cls_logits = output[:, 0:2]
    reg_logits = output[:, 2:4]
    reg_right = output[:, 2]
    reg_left = output[:, 3]

    return  cls_logits, reg_logits, reg_left, reg_right


def cls_loss(cls_logits, labels):

    return F.cross_entropy(cls_logits, labels.squeeze())


def loc_loss(reg_logits, offsets, labels):
    loc_diff = torch.sum((offsets - reg_logits) ** 2, 1)
    return torch.dot(labels.view_as(loc_diff), loc_diff)







