import torch
import torch.nn.functional as F
from torch.autograd import Variable


def EMD_L2(predictions, targets, needSoftMax=True):
    #TODO: checked run but not checked correctness!
    if needSoftMax:
        predictions = F.softmax(predictions, dim=1)

    predictions = torch.cumsum(predictions, dim=1)
    targets = torch.cumsum(targets, dim=1)
    lossvalue = torch.norm(predictions - targets, p=2).mean()
    return lossvalue
    # elm_loss = (predictions - targets)**2
    # batch_size = predictions.size(0)
    # return  torch.sum(elm_loss)/batch_size

def Simple_L2(predictions, targets, needSoftMax=True):
    #TODO: checked run but not checked correctness!
    if needSoftMax:
        predictions = F.softmax(predictions, dim=1)

    # predictions = torch.cumsum(predictions, dim=1)
    # targets = torch.cumsum(targets, dim=1)
    lossvalue = torch.norm(predictions - targets, p=2).mean()
    return lossvalue


def to_one_hot(y, n_dims=None, useCuda=True):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    if useCuda:
        y_one_hot=y_one_hot.cuda()

    y_shape = [i for i in y.shape]
    y_shape.append(-1)
    
    y_one_hot = y_one_hot.view(*y_shape)
    return Variable(y_one_hot, requires_grad=False) if isinstance(y, Variable) else y_one_hot

def WeightedMSE(predictions, targets, weights):
    #TODO: check if there are error heres
    return torch.dot(weights.view_as(predictions), (predictions - targets) ** 2)


def MSEloss(predictions, targets):
    return torch.sum((predictions - targets.view_as(targets))**2)


def ClsLocLoss(predictions, targets, overlaps):
    # adapted from Shou CVPR2016
    # caffe version defined in
    # https://github.com/zhengshou/scnn/blob/master/C3D-v1.0/C3D_overlap_loss/src/caffe/layers/softmax_loss_layer.cpp
    # loss_overlap += 0.5 * (prob_data[i * dim + static_cast < int > (label[i])] / static_cast < float > (std
    #                                                                      ::sqrt(std::sqrt(std::sqrt(overlap[i])))))*(
    #             prob_data[i * dim + static_cast < int > (label[i])] / static_cast < float > (std
    #                                                                 ::sqrt(std::sqrt(std::sqrt(overlap[i]))))) - 0.5;
    target_weights = (targets!=0).float()
    loss = (predictions**2/ overlaps.pow(0.25) - targets**2)
    loss = torch.sum(loss * target_weights)
    return loss


def ClsLocLoss2_OneClsRegression(predictions, targets, overlaps):
    # following the previous loss, but modified for our use case
    # overlap with ground truth is 0, we set overlap to 1 to avoid divison with 0, but weight them 0

    # target_weights = (overlaps!=0).float()
    loss = (predictions / overlaps - targets.float())**2
    loss = torch.sum(loss)
    return loss

def ClsLocLoss2_OneClsRegression_v2(predictions, overlaps):
    # following the previous loss, but modified for our use case
    # overlap with ground truth is 0, we set overlap to 1 to avoid divison with 0, but weight them 0

    # target_weights = (overlaps!=0).float()
    loss = (predictions - overlaps)**2
    loss = torch.sum(loss)
    return loss



def ClsLocLoss_MultiClass(predictions, targets, overlaps, useSoftMax=True):
    if useSoftMax:
        predictions = F.softmax(predictions, dim=1)
    overlaps = overlaps.view_as(targets)

    target_weights = (overlaps!=0).float()
    overlaps[overlaps==0]=1

    loss = (predictions.gather(dim=1, index=targets.long())/(overlaps.pow(0.125)) - 1)**2
    loss = torch.sum(loss * target_weights)
    return loss


def ClsLocLoss_Regression(predictions, targets, overlaps, thres=0.7, useSoftMax=True):
    overlaps = overlaps.view_as(targets)
    targets[overlaps<thres]=0
    target_weights = (overlaps!=0).float()
    overlaps[overlaps==0]=1

    loss = (predictions/(overlaps.pow(0.125)) - targets)**2
    loss = torch.sum(loss * target_weights)
    return loss

if __name__ == '__main__':
    import numpy as np
    from torch.autograd import Variable

    predictions = torch.rand(2,5)
    targets = torch.zeros(*predictions.size())
    targets[0,1] = 1
    targets[1,2] = 1

    predictions = Variable(predictions, requires_grad=True)
    targets = Variable(targets, requires_grad=False)
    loss = EMD_L2(predictions, targets, needSoftMax=True)
    loss.backward()
    print "DB"