# -*- coding: utf-8 -*-
"""
Created on 2017/5/4

@author: Naiive
显示测试结果
"""
# from time import time
# from PIL import Image
# import glob
import numpy as np
# import sys
# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from get_eigenvalue import *
from get_svm import *

pic = get_eigenvalue()
print (pic.get_pca_eigen())
print("------------------------------------------")
x_train, x_test, y_train, y_test = pic.get_data()#存放图像降维后的灰度特征向量
x_train1, x_test1, y_train1, y_test1 = pic.get_colordata()#存放图像降维前的彩色特征向量
x_train0, x_test0, y_train0, y_test0 = pic.get_graydata()#存放图像降维前的灰度特征向量
img_size = pic.get_img_size()#记录各图片的h，w
target_n = pic.get_target_n()#存放图片名字，下面函数的target存放0，1，2.。这些数字，对应这里的序号

# test_pred, precision_average = SVM1("rbf", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], pic)
# #test_pred存放预测结果，对应y_test
# print ("预测的准确率为：", precision_average)
# result = []
# for i in range(len(test_pred)):
#     result.append((y_test[i], test_pred[i]))

#绘制预测准确度的曲线
def show_pecise():
    kernel_to_test = ['rbf', 'poly', 'sigmoid']

    for kernel_name in kernel_to_test:
        x_label = np.linspace(0.0001, 1, 100)
        y_label = []
        for i in x_label:
            a, b = SVM2(kernel_name, [i], pic)
            y_label.append(b)
        plt.plot(x_label, y_label, label=kernel_name)

    plt.xlabel("Gamma")
    plt.ylabel("Precision")
    plt.title('Different Kernels Contrust')
    plt.legend()
    plt.show()
show_pecise()
