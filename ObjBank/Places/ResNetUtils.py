import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch

import PtUtils.cuda_model as cuda_model
import os
import shutil

def Res50Places_val_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

#
# def getRes50PlacesModel(eval=False, gpu_id=None, multiGpu=False):
#     model_url = 'http://places2.csail.mit.edu/models_places365/whole_resnet50_places365-c66f227f.pth.tar'
#     model = model_zoo.load_url(model_url)
#
#     model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
#     if eval:
#         model.eval()
#     return model



def getRes50PlacesModel(eval=False, gpu_id=None, multiGpu=False):
    weight_file = 'whole_resnet50_places365.pth.tar'
    model_url = 'http://places2.csail.mit.edu/models_places365/' + weight_file
    saved_model_path = os.path.join(os.path.expanduser('~'), 'datasets/PretrainedModels', 'Places', weight_file)

    if not os.access(saved_model_path, os.W_OK):
        os.system('wget ' + model_url)
        shutil.move(weight_file, saved_model_path)

    useGPU = cuda_model.ifUseCuda(gpu_id, multiGpu)
    if useGPU:
        model = torch.load(saved_model_path)
    else:
        model = torch.load(saved_model_path, map_location=lambda storage,
                                                                 loc: storage)  # allow cpu

    model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
    if eval:
        model.eval()
    return model