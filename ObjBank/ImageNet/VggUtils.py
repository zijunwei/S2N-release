import torchvision.transforms as transforms
import torchvision.models as models

import PtUtils.cuda_model as cuda_model

def VGGImageNet_val_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def getVGGImageNetModel(eval=False, gpu_id=None, multiGpu=False):
    # before it was using bn, now change to 16 since bn is out of save
    # model = models.vgg16_bn(pretrained=True)

    model = models.vgg16(pretrained=True)
    model = cuda_model.convertModel2Cuda(model, gpu_id, multiGpu)
    if eval:
        model.eval()
    return model