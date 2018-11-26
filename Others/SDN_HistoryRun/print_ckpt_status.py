#Update: Compared to v4, now the layer is bidirectional and set some learnable init states for encoder

import os, sys
project_root = os.path.join(os.path.expanduser('~'), 'Dev/NetModules')
sys.path.append(project_root)

import torch
print(torch.__version__)

import glob



ckpt_dir = '/home/zwei/Dev/NetModules/ckpts/SDN_mnist_EMD_hmatch-assgin0.75-alpha0.1000-dim512-dropout0.5000-seqlen100-ckpt'

ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.tar'))
ckpt_files.sort()
for s_ckpt_file in ckpt_files:
    ckpt_filename = os.path.basename(s_ckpt_file)
    checkpoint = torch.load(s_ckpt_file, map_location=lambda storage, loc: storage)
    train_iou = checkpoint['IoU']
    train_tloss = checkpoint['loss']
    train_cls_loss = checkpoint['cls_loss']
    train_loc_loss = checkpoint['loc_loss']

    print(" {}, total loss: {:.04f},\t cls_loss: {:.04f},\t loc_loss: {:.04f},"
          " \tcurrent iou: {:.04f}".format(ckpt_filename, train_tloss, train_cls_loss, train_loc_loss, train_iou))