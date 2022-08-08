import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torchvision import transforms
from tqdm import tqdm
import train
from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    classes = ['benign', 'malignant']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def pred(model, data_loader, device):

    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    pred_all = []
    label_names = ['benign', 'malignant']
    label_all = []
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        red_classes = torch.max(pred, dim=1)[1]
        pred_all.append(red_classes.cpu().numpy())
        label_all.append(labels.cpu().numpy())
    pred_all = [i for item in pred_all for i in item]
    labels_all = [i for item in label_all for i in item]
    print(metrics.classification_report(labels_all, pred_all, labels=range(2), target_names=label_names, digits=4))
    cm = metrics.confusion_matrix(labels_all, pred_all, labels=range(2))
    print(cm)
    # plot_confusion_matrix(cm, savename='./png/vit_9.png')


if __name__ == '__main__':
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize([224,224]),
                                   # transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=2,has_logits=False).to(device)
    weight = r'./weights/17个序列/vit-9.pth'
    val_images_path, val_images_label = read_split_data(r'E:\Data\Mydataset\all\crop\alls\fold4')[2:]
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    batch_size=16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    model.load_state_dict(torch.load(weight))
    pred(model,val_loader,device)