import torch.nn as nn
import torch.nn.functional as F
import totrchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import torch 
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

net=models.resnet34(pretrained=True)

net.eval()

im_orig=Image.open('img_path')
mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]

im=transofrms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std)
])(im_orig)


r,loop_i,label_orig,label_pert,pert_image=deepfool(im,net)
labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')


str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)