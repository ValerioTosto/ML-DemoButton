import torch
from torch import nn
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16


def get_model_squeezenet(pretrained):
    model = squeezenet1_0(pretrained)
    num_class = 4
    model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_class
    return model

def get_model_vgg(pretrained):
    model = vgg16(pretrained)
    num_class = 4
    model.classifier[6] = nn.Linear(4096, num_class)
    return model
