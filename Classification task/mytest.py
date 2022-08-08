

# import pretrainedmodels.models as models
#
# use_model = models.senet.se_resnet101(num_classes=1000,pretrained='imagenet')
import torch

import Models
import fusionModels
import train
from VIT_model import vit_base_patch16_224_in21k as vit

# content = torch.load('./model/thynet/resnet101_400.pth')
# print(content.keys())
from torch import nn


if __name__ == '__main__':
    model = Models.Resnet50(num_classes=2)
    vits = vit(num_classes=2,has_logits=False).cuda()
    fusion = fusionModels.FusionM(num_classes=2)

    # model.conv1 = nn.Conv2d(1,64,kernel_size=7,padding=3,stride=2,bias=False)
    model = torch.nn.DataParallel(model).cuda()
    fusion = torch.nn.DataParallel(fusion).cuda()

    path = './model/ThyNet/fusion/91.78%_2层se_vit7_all.pth'
    f_path = './model/ThyNet/全部17个序列/88.94%.pth'
    s_path = './model/ThyNet/全部17个序列/senet50.pth'
    v_path = 'model/ThyNet/全部17个序列/vit.pth'
    # content = torch.load(path)
    # print(content.keys())
    z_path = r'E:\代码\ThyNet-main\model\ThyNet\转移性质\resnet50_DCE_F4_12.pth'
    fusion.load_state_dict(torch.load(f_path))
    model.load_state_dict(torch.load(z_path))
    vits.load_state_dict(torch.load(v_path))
    train.test(model)



