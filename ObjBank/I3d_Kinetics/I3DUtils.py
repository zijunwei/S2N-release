import os

import torch

import PtUtils.cuda_model as cuda_model
from ObjBank.I3d_Kinetics.I3D_Pytorch import I3D
from torchvision import transforms
IMAGE_SIZE = 224
NUM_CLASSES = 400
SAMPLE_VIDEO_FRAMES = 64

def K_I3D_val_transform():
    pass


def getK_I3D_RGBModel(eval=False, gpu_id=None, multiGpu=False):
    weight_file = 'rgb_imagenet.pkl'
    saved_model_path = os.path.join(os.path.expanduser('~'), 'datasets/PretrainedModels', 'Kinetics', weight_file)

    model = I3D(input_channel=3)
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)
    model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
    if eval:
        model.eval()
    return model


def getK_I3D_FlowModel(eval=False, gpu_id=None, multiGpu=False):
    weight_file = 'flow_imagenet.pkl'
    saved_model_path = os.path.join(os.path.expanduser('~'), 'datasets/PretrainedModels', 'Kinetics', weight_file)

    model = I3D(input_channel=2)
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)
    model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
    if eval:
        model.eval()
    return model


def simple_I3D_transform(image_size=None):
    operations = [] if image_size is None else [transforms.Resize((image_size, image_size))]
    operations.append(transforms.ToTensor())
    operations.append(transforms.Normalize(mean=[-1., -1., -1.], std=[2., 2., 2.]))
    transform = transforms.Compose(operations)
    return transform