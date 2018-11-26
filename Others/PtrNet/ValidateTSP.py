"""
Pytorch implementation of Pointer Network.
http://arxiv.org/pdf/1506.03134v1.pdf.
"""
import os
import sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from PointerNetLSTMDecoder import PointerNet
from PtrNet.data_generator.TSPDataGenerator import TSPDataset
from PtUtils import cuda_model
import Metrics

parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=256, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument("--gpu_id", default='0', type=str)
parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# TSP
parser.add_argument('--nof_points', type=int, default=10, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=2, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=6, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

args = parser.parse_args()

use_cuda = cuda_model.ifUseCuda(args.gpu_id, args.multiGpu)


model = PointerNet(args.embedding_size,
                   args.hiddens,
                   args.nof_lstms,
                   args.dropout,
                   args.bidir)
model = cuda_model.convertModel2Cuda(model, gpu_id=args.gpu_id, multiGpu=args.multiGpu)

dataset = TSPDataset(args.train_size,
                     args.nof_points)


dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=4)




CCE = torch.nn.CrossEntropyLoss()

model_optim = optim.RMSprop(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=args.lr)


losses = []

for epoch in range(args.nof_epoch):
    batch_loss = []
    # lr_scheduler.step(epoch)
    # print('Epoch\t{:d}\t LR lr {:.5f}'.format(epoch, )

    iterator = tqdm(dataloader, unit='Batch')

    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch + 1, args.nof_epoch))

        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        if use_cuda:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)
        accuracy = Metrics.accuracy_topN(p.data, target_batch.data)
        o = o.contiguous().view(-1, o.size()[-1])

        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)

        losses.append(loss.data[0])
        batch_loss.append(loss.data[0])

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()
        iterator.set_postfix_str('lr={:.6f}\tloss={:.6f}\tacc={:.4f}'.format(model_optim.param_groups[0]['lr'],loss.data[0], accuracy[0]))

    iterator.set_postfix(loss=np.average(batch_loss))