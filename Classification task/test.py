import torch
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
import Models
import numpy as np
import os
#from skimage import io, transform
import time, os, json
#from logutil import initlogging
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--resnet_model_path', type=str, default='', help='The trained resnet model path')
parser.add_argument('--densenet_model_path', type=str, default='', help='The trained densenet model path')
parser.add_argument('--resnext_model_path', type=str, default='', help='The trained resnext model path')
parser.add_argument('--imgpath', type=str, default=r'E:\Data\转移性质\val\0', help='The test image path')
parser.add_argument('--gpu', type=str, default='0', help='using gpu id')
parser.add_argument('--num_class', type=int, default=2, help='Number of categorized tasks')
arg = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]
def get_pytorch_image(name):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std, inplace=True)
    ])
    image = Image.open(name)

    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.cuda()
    return image


def load_dict(model, model_dir):
    model = torch.nn.DataParallel(model).cuda()
    if 'rxt101' in model_dir:
        pre_dict = torch.load(model_dir)
        pre_dict = {k: v for k, v in pre_dict.items() if 'last_linear' not in k}
        model.load_state_dict(pre_dict,strict=False)
    else:
        model.load_state_dict(torch.load(model_dir),strict=False)
    return model.cuda()


def load_model():
    model_dict = {'res101': Models.Resnet101(num_classes = 2 ),
                  'rxt101': Models.Resnext101(num_classes = 2 ),
                  'den201': Models.Densnet201(num_classes = 2 )}
    model_dir = {'res101': arg.resnet_model_path,
                 'rxt101': arg.resnext_model_path,
                 'den201': arg.densenet_model_path,
                 }
    model_res = load_dict(model_dict['res101'], model_dir['res101'])
    model_rxt = load_dict(model_dict['rxt101'], model_dir['rxt101'])
    model_den = load_dict(model_dict['den201'], model_dir['den201'])
    return model_res, model_rxt, model_den


def output(model, data):
    model.eval()
    s_output = model(data)
    M_possibility = float(F.softmax(s_output,dim=1).data.cpu().numpy()[0][1])
    pred = s_output.data.max(1)[1]  # get the index of the max utils-probability


    return pred




# def get_result(image):
#     result = output(model_res, image)  + output(model_rxt, image) + output(model_den, image)
#
#     if result >= 2:
#         return 1
#     else:
#         return 0


# model_res, model_rxt, model_den = load_model()
model = Models.Resnet50(num_classes=2)
model = torch.nn.DataParallel(model).cuda()
path = './model/ThyNet/转移性质/resnet50_28_Breast.pth'
model.load_state_dict(torch.load(path))
pred_all = []
for img in glob.glob(arg.imgpath+'/*.jpg'):
    py_image = get_pytorch_image(img)
    pred = output(model,py_image)
    pred_all.append(pred.cpu().numpy())
print(pred_all)
# result = get_result(py_image)
# print(arg.imgpath, result)
