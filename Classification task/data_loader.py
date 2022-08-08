from torchvision import datasets, transforms
import torch
import numpy as np
import os

norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]
norm_mean1 = [0.485,]
norm_std1 = [0.229,]
def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
         # transforms.Grayscale(1),
         transforms.Resize([256,256]),##重置图像分辨率
         # transforms.RandomRotation(15),##以指定的角度选装图片。（-15°，15°）
         transforms.ColorJitter(),##随机修改亮度、对比度和饱和度
         transforms.RandomCrop(224),##依据给定的size随机裁剪
         transforms.RandomHorizontalFlip(),##以给定的概率随机水平翻折PIL图片。默认概率是0.5
         transforms.RandomVerticalFlip(),##依据概率p对PIL图片进行垂直翻转
         transforms.ToTensor(),#将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
         transforms.Normalize(norm_mean,norm_std,inplace=True)
         ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir),transform=transform)##将特征与标签进行组合
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):



    transform = transforms.Compose([
         # transforms.Grayscale(1),
         transforms.Resize([224,224]),
         # transforms.Resize(256),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(norm_mean,norm_std,inplace=True)
         ])##训练集不做数据增强操作
    data = datasets.ImageFolder(root=os.path.join(root_path, dir),transform=transform)
    # print(list(data.imgs))
    names = list(map(lambda x: os.path.basename(x[0]), list(data.imgs)))#os.path.basename 返回路径最后的文件名
    label = list(map(lambda x: x[1], list(data.imgs)))
    # print(names, label)
    # for name, label in data.imgs:
    #     print(name, label)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader, names, label


if __name__ == '__main__':
    train_loader= load_training('data','train',batch_size=64,kwargs={})
    i=0
    for data,label in train_loader:
       print(data.shape)