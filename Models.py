import pretrainedmodels.models
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torchvision import models
import torch.nn.functional as F
import pretrainedmodels.models as premodels
from pretrainedmodels.models.resnext_features import resnext101_32x4d_features



from pretrainedmodels.models.resnext_features import resnext101_64x4d_features
from lambda_networks import λLayer


class vgg16(nn.Module):
    def __init__(self, num_classes=2):
        super(vgg16, self).__init__()
        model_vgg = models.vgg16(pretrained=True)
        self.features = model_vgg.features
        self.avgpool = model_vgg.avgpool
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.classifer(x)
        return x


class Resnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet18, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__features = model_resnet18.fc.in_features
        self.fc = nn.Linear(self.__features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Resnet50(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet50, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__features = model_resnet50.fc.in_features
        self.fc = nn.Linear(self.__features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Resnet101(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet101, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features
        self.fc = nn.Linear(2048, num_classes)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

    def output_num(self):
        return self.__in_features

class Resnet152(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet152, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.__in_features = model_resnet152.fc.in_features
        self.fc = nn.Linear(2048, num_classes)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

class Densenet169(nn.Module):
    def __init__(self, num_classes=2):
        super(Densenet169, self).__init__()
        model_densenet169 = models.densenet169(pretrained=True)
        self.features = model_densenet169.features
        self.fc = nn.Linear(1664, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.fc(out)
        return out


class Densnet201(nn.Module):
    def __init__(self, num_classes=2):
        super(Densnet201, self).__init__()
        model_densenet201 = models.densenet201(pretrained=True)
        self.features = model_densenet201.features
        self.fc = nn.Linear(1920, num_classes)
        self.relu_out = 0

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        self.relu_out = out
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.fc(out)
        return out

    def cam_out(self):
        return self.relu_out


class ResNeXt101_32x4d(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.num_classes = num_classes
        self.features = resnext101_32x4d_features
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(2048, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class Resnext101(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnext101, self).__init__()
        use_model = ResNeXt101_32x4d()
        use_model.load_state_dict(torch.load('./model/resnext101_32x4d-29e315fa.pth'))
        self.features = use_model.features
        self.avg_pool = use_model.avg_pool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Senet101(nn.Module):
    def __init__(self, num_classes=2):
        super(Senet101, self).__init__()
        model_se = premodels.se_resnet101()
        self.layer0 = model_se.layer0
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        self.layer3 = model_se.layer3
        self.layer4 = model_se.layer4
        self.avgpool = model_se.avg_pool
        self.dropout = model_se.dropout
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Senet50(nn.Module):
    def __init__(self,num_classes=2):
        super(Senet50, self).__init__()
        model_se = premodels.se_resnet50()
        self.layer0 = model_se.layer0
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        self.layer3 = model_se.layer3

        self.layer4 = model_se.layer4
        self.avgpool = model_se.avg_pool
        self.dropout = model_se.dropout
        self.fc = nn.Linear(2048, num_classes)
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.lambdas = layer
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.lambdas(self.relu(self.bn1(self.conv1(x)))))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


layer = λLayer(dim=32,
               dim_out=32,
               n=64,
               dim_k=16,
               heads=4,
               dim_u=1)

# model = Senet101(num_classes=2)
# model = model.cuda()
# summary(model,(3,224,224))
