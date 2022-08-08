# -*- coding: utf-8 -*-
"""
Created on 2017/5/4

@author: Naiive
显示测试结果:比较结果
"""
# from time import time
# from PIL import Image
# import glob
# import numpy as np
# import sys
# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from get_eigenvalue import *
from get_svm import *
from time import time

pic = get_eigenvalue()
print (pic.get_pca_eigen())
print("------------------------------------------")
x_train, x_test, y_train, y_test = pic.get_data()#存放图像降维后的灰度特征向量
x_train1, x_test1, y_train1, y_test1 = pic.get_colordata()#存放图像降维前的彩色特征向量
x_train0, x_test0, y_train0, y_test0 = pic.get_graydata()#存放图像降维前的灰度特征向量
img_size = pic.get_img_size()#记录各图片的h，w
target_n = pic.get_target_n()#存放图片名字，下面函数的target存放0，1，2.。这些数字，对应这里的序号

clf = SVM1("rbf", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], pic)
# 预测图片
print("------------------------------------------")
print("正在预测图片...")
t0 = time()
test_pred = clf.predict(x_test)
print("完成预测，共计 %0.3f秒" % (time() - t0))
print("------------------------------------------")
print("预测结果如下：")
print(classification_report(y_test, test_pred, target_names=target_n))
n_classes = pic.get_nclasses()
print(confusion_matrix(y_test, test_pred, labels=range(n_classes)))
print("------------------------------------------")
prediction = 0
for i in range(len(test_pred)):
    if test_pred[i] == y_test[i]:
        prediction += 1
precision_average = prediction / len(y_test)

#test_pred存放预测结果，对应y_test
print ("预测的准确率为：", precision_average)
result = []
for i in range(len(test_pred)):
    result.append({"target:":target_n[y_test[i]], "predict:":target_n[test_pred[i]]})
for i in range(100):
    print (result[i])

#比较预测值和实际值
def plot_gallery(images, titles, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(2.5 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35, wspace = .35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        #plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        # image = images[i].reshape(h, w)
        # img = []
        # for j in range(len(image)):
        #     img.append(list(image[j]))
        # #print(img[:5])
        plt.imshow(images[i])
        #plt.imshow(images[i])
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    if y_pred[i] == y_test[i]:
        pre = "(√)"
    else:
        pre = '(×)'
    return 'predicted: %s%s\ntrue:      %s' % (pred_name, pre, true_name)
def show_prediction():
    prediction_titles = [title(test_pred, y_test, target_n, i)
                         for i in range(len(test_pred))]

    plot_gallery(x_test1, prediction_titles)

    # plot the gallery of the most significative eigenfaces

    #eigenface_titles = ["eigenface %d" % i for i in range(len(eigenfaces))]
    #plot_gallery(eigenfaces, eigenface_titles, h = 30, w = 18)

    plt.show()
show_prediction()


