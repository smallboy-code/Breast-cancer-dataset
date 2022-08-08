
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from torch import nn
import torch.nn.functional as F
import Models
import train
from VIT_model import vit_base_patch16_224_in21k as vit



if __name__ == '__main__':
    path = './model/ThyNet/senet50_17.pth'
    # path = './model/ThyNet/全部17个序列/senet50.pth'
    vit_path = './model/ThyNet/vit_7层.pth'
    # vit_path = './model/ThyNet/全部17个序列/vit.pth'
    senet = Models.Senet50(num_classes=2)
    senet.load_state_dict({k.replace('module.','',1):v for k,v in torch.load(path).items()})
    vit_7 = vit(num_classes=2,has_logits=False)
    vit_7.load_state_dict(torch.load(vit_path))
    x1 = train.source_loader
    x2 = train.target_val_loader
    senet.eval()
    vit_7.eval()
    pred_all=[]
    cuda = torch.cuda.is_available()
    possbilitys = None
    for X_test,y_test in x2:
        if cuda:
            X_test,y_test = X_test.cuda(),y_test.cuda()
            senet = senet.cuda()
            vit_7 = vit_7.cuda()
        out_se = senet(X_test)
        out_vit = vit_7(X_test)
        se_possibility = F.softmax(out_se,dim=1)
        vit_possibilty = F.softmax(out_vit,dim=1)
        # fu_possibility = torch.add(se_possibility,vit_possibilty)/2
        max_possibility = torch.maximum(se_possibility,vit_possibilty)
        if possbilitys is None:
            possbilitys = max_possibility.cpu().data.numpy()
        else:
            possbilitys = np.append(possbilitys, max_possibility.cpu().data.numpy(), axis=0)
        pred = max_possibility.data.max(1)[1]
        pred_all.append(pred.cpu().numpy())
    pred_all = np.hstack(pred_all)
    # print(possbilitys)
    y_test = np.loadtxt('./features/y_test.txt', delimiter=',')
    target_name = ['benign', 'malignant']
    cm = confusion_matrix(y_test, pred_all)
    label_onehot = np.eye(2)[np.array(y_test).astype(np.int32).tolist()]
    fpr, tpr, thresholds = roc_curve(label_onehot.ravel(), possbilitys.ravel(), pos_label=1)
    auc_2 = roc_auc_score(y_test, possbilitys[:,1])
    # print(auc_2)
    print('auc: ',auc(fpr,tpr))
    print(classification_report(y_test, pred_all, target_names=target_name, digits=4))
    print(cm)