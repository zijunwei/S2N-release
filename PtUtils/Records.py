import os
import torch
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    save_directory = os.path.dirname(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_directory, 'model_best.pth.tar'))