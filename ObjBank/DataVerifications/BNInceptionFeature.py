import numpy as np
import os, sys

# TODO: No longer working now since src_file2 is deleted permanately

filename = 'Bus_in_Rock_Tunnel.npy'
src_file1 = '/home/zwei/datasets/SumMe/features/ImageNet/BNInception/{:s}'.format(filename)
src_file2 = '/home/zwei/datasets/SumMe/features/ImageNet/BNIncpetion/{:s}'.format(filename)

data1 = np.load(src_file1)
data2 = np.load(src_file2)
# checked equal
print "DEBUG"