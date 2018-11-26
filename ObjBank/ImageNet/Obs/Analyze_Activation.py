import numpy as np


def CheckActivation(activation, activation_name):
    abs_activation = np.abs(activation)
    mean_abs_activation = np.mean(np.mean(abs_activation, axis=2), axis=2)
    mean_abs_activation = np.squeeze(mean_abs_activation)
    max_activation = np.max(mean_abs_activation)
    percentile = 0.1
    thres = percentile*max_activation
    sparsity = sum(mean_abs_activation > thres)*1. / len(mean_abs_activation)
    print '{:s}\t{:.04f}'.format(activation_name, sparsity)
