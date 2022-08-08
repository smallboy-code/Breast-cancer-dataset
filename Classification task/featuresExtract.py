import numpy as np
import torch
from torch import nn

import Models
import train
from VIT_model import vit_base_patch16_224_in21k as vit
from tkinter import _flatten




if __name__ == '__main__':
    # path = './model/ThyNet/senet50_17.pth'
    path = './model/ThyNet/全部17个序列/senet50.pth'
    # vit_path = './model/ThyNet/vit_7层.pth'
    vit_path = './model/ThyNet/全部17个序列/vit.pth'
    senet = Models.Senet50(num_classes=2)
    senet.load_state_dict({k.replace('module.','',1):v for k,v in torch.load(path).items()})
    vit_6 = vit(num_classes=2,has_logits=False)
    vit_6.load_state_dict(torch.load(vit_path))
    del senet.fc
    del vit_6.head
    x1 = train.source_loader
    x2 = train.target_val_loader
    senet.fc = nn.Linear(2048, 2048)
    nn.init.eye_(senet.fc.weight)
    senet.eval()
    for param in senet.parameters():
        param.requires_grad = False
    vit_6.head = nn.Linear(768, 768)
    nn.init.eye_(vit_6.head.weight)
    vit_6.eval()
    for param in vit_6.parameters():
        param.requires_grad = False
    x_train_list,y_train_list=[],[]
    x_test_list,y_test_list=[],[]
    cuda = torch.cuda.is_available()
    for X_train,y_train in x1:
        if cuda:
            X_train,y_train = X_train.cuda(),y_train.cuda()
            senet = senet.cuda()
            vit_6 = vit_6.cuda()
        out_se = senet(X_train)
        out_vit = vit_6(X_train)
        a = torch.cat((out_se,out_vit),1).cpu().data.numpy()
        x_train_list.append(a)
        y_train_list.append(y_train.cpu().data.numpy())
    for X_test,y_test in x2:
        if cuda:
            X_test,y_test = X_test.cuda(),y_test.cuda()
            senet = senet.cuda()
            vit_6 = vit_6.cuda()
        out_se = senet(X_test)
        out_vit = vit_6(X_test)
        a = torch.cat((out_se, out_vit), 1).cpu().data.numpy()
        x_test_list.append(a)
        y_test_list.append(y_test.cpu().data.numpy())
    # mat_train = np.array(x_train_list,dtype=object)
    # mat2_train = np.array(y_train_list,dtype=object)
    # mat_test = np.array(x_test_list,dtype=object)
    # mat2_test = np.array(y_test_list,dtype=object)

    mat_stack_train = np.vstack(x_train_list)
    mat2_stack_train = np.vstack(y_train_list)
    mat_stack_test = np.vstack(x_test_list)
    mat2_stack_test = np.hstack(y_test_list)

    np.savetxt('./features/x_train_all.txt', mat_stack_train, delimiter=',')
    np.savetxt('./features/y_train_all.txt', mat2_stack_train,fmt='%d',delimiter=',')
    np.savetxt('./features/x_test_all.txt', mat_stack_test, delimiter=',')
    np.savetxt('./features/y_test_all.txt',mat2_stack_test,fmt='%d',delimiter=',')
    # t = ''
    # with open('./features/y_test.txt', 'w') as q:
    #     for i in y_test_list:
    #         for e in range(len(i)):
    #             t = t + str(i[e]) + ','
    #         q.write(t.strip(','))
    #         q.write('\n')
    #         t = ''

# y = senet(x).cpu()
# y = y.data.numpy()
# np.savetxt('./features/senet.txt', y, delimiter=',')
