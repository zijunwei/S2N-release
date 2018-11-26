import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import matplotlib.gridspec as gridspec
import PyUtils.dir_utils as dir_utils
import progressbar

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import torchvision.transforms as T
import os
def show_images(images, save_path):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    ax = plt.gca()
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.imshow(images.reshape([sqrtimg, sqrtimg]))
    fig.savefig(save_path, bbox_inches='tight',pad_inches=0, transparent = True)
    plt.close(fig)
    # gs = gridspec.GridSpec(sqrtn, sqrtn)
    # gs.update(wspace=0.05, hspace=0.05)





# answers = np.load('gan-checks-tf.npz')
# print('Done')

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 100
batch_size = 1

# mnist_train = dset.MNIST('./data/MNIST_data', train=True, download=True,
#                            transform=T.ToTensor())
# loader_train = DataLoader(mnist_train, batch_size=batch_size,
#                           sampler=ChunkSampler(NUM_TRAIN, 0))

mnist_val = dset.MNIST('./data/MNIST_data', train=False, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size)

image_save_dir = dir_utils.get_dir('./data/MNIST_data/images')
pbar = progressbar.ProgressBar(max_value=len(loader_val))
for s_idx, s_data in enumerate(loader_val):
    pbar.update(s_idx)
    s_digit = s_data[0]
    s_label = s_data[1]
    s_digit = s_digit.view(batch_size, 784).numpy()
    save_name = '{:d}-{:06d}.png'.format(s_label.item(), s_idx)
    save_path = os.path.join(image_save_dir, save_name)
    show_images(s_digit, save_path)
    # print("DB")
# imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()
# show_images(imgs)