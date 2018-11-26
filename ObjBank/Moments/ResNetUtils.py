import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.models as models

import PtUtils.cuda_model as cuda_model


def M_Res50_val_transform():
        """Load the image transformer."""
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf


def getM_Res50Model(eval=False, gpu_id=None, multiGpu=False, categories=339):
    weight_file = 'moments_RGB_resnet50_imagenetpretrained.pth.tar'

    saved_model_path = os.path.join(os.path.expanduser('~'), 'datasets/PretrainedModels', 'Moments', weight_file)

    if not os.access(saved_model_path, os.W_OK):
        weight_url = 'http://moments.csail.mit.edu/moments_models/' + weight_file
        os.system('wget ' + weight_url)
        shutil.move(weight_file, saved_model_path)

    model = models.__dict__['resnet50'](num_classes=categories)
    useGPU = cuda_model.ifUseCuda(gpu_id, multiGpu)
    if useGPU:
        checkpoint = torch.load(saved_model_path)
    else:
        checkpoint = torch.load(saved_model_path, map_location=lambda storage,
                                                                 loc: storage)  # allow cpu

    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

    model.load_state_dict(state_dict)
    model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
    if eval:
        model.eval()
    return model

