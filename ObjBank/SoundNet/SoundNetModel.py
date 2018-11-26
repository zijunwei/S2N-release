import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.serialization import load_lua

from torch.autograd import Variable
import os

torch_model_path = os.path.join(os.path.expanduser('~'), 'datasets/Models/SoundNet', 'soundnet8_final.t7')



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 64, stride=2, padding=32)
        self.pool1 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(16, 32, 32, stride=2, padding=16)
        self.pool2 = nn.MaxPool1d(8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(32, 64, 16, stride=2, padding=8)
        self.conv4 = nn.Conv1d(64, 128, 8, stride=2, padding=4)
        self.conv5 = nn.Conv1d(128, 256, 4, stride=2, padding=2)
        self.pool5 = nn.MaxPool1d(4, stride=1, padding=0)
        self.conv6 = nn.Conv1d(256, 512, 4, stride=2, padding=2)
        self.conv7 = nn.Conv1d(512, 1024, 4, stride=2, padding=2)
        self.conv8_1 = nn.Conv1d(1024, 1000, 4, stride=2, padding=0)
        self.conv8_2 = nn.Conv1d(1024, 401, 4, stride=2, padding=0)
        self.fc1 = nn.Linear(859000, 1000)
        self.fc2 = nn.Linear(344459, 365)

    def forward(self, input_wav):
        x = self.pool1(F.relu(nn.BatchNorm1d(16)(self.conv1(input_wav))))
        x = self.pool2(F.relu(nn.BatchNorm1d(32)(self.conv2(x))))
        x = F.relu(nn.BatchNorm1d(64)(self.conv3(x)))
        x = F.relu(nn.BatchNorm1d(128)(self.conv4(x)))
        x = self.pool5(F.relu(nn.BatchNorm1d(256)(self.conv5(x))))
        x = F.relu(nn.BatchNorm1d(512)(self.conv6(x)))
        x = F.relu(nn.BatchNorm1d(1024)(self.conv7(x)))
        x_object = Flatten()(F.relu(self.conv8_1(x)))
        x_place = Flatten()(F.relu(self.conv8_2(x)))
        x_object = self.fc1(x_object)
        x_place = self.fc2(x_place)
        y = [x_object, x_place]
        return y



if __name__ == '__main__':
    pass

    # m = torch.load('xxxx.t7')
    # m = m:float()
    # torch.save('xxxx.cpu.t7', m)
