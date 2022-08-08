# -*- coding: utf-8 -*-
"""
Created on 2017/5/4

@author: Naiive
获取降维后的图片特征向量和target
"""
from time import time
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd


#---------------------------------------------#
#获取图片的特征向量和target
__all__ = ['get_eigenvalue']

class get_eigenvalue:
    """
    input:存放图片的总文件夹
    """
    target_n = []#存放图片名字，下面函数的target存放0，1，2.。这些数字，对应这里的序号
    img_size = []#记录各图片的h，w
    pic_data_rgb = []#存放图片集的彩色信息(r,g,b)
    pic_data_rgb_T = []
    pic_data_gray = []#存放图片集的灰度信息
    pic_data_gray_test = []
    target = []#存放所有样本对应的分类数
    target_t = []#测试样本对应的分类数
    n_classes = 0#存放分类总数
    x_train0, x_test0, y_train0, y_test0 = [], [], [], []#存放图像降维前的灰度特征向量
    x_train1, x_test1, y_train1, y_test1 = [], [], [], []#存放图像降维前的彩色特征向量
    eigenfaces = []#存放特征脸
    x_train, x_test, y_train, y_test = [], [], [], [] #存放图像降维后的灰度特征向量
    pca = None

    #得到用于训练和验证的特征向量
    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
    def get_target_n(self):
        return self.target_n
    def get_img_size(self):
        return self.img_size
    def get_colordata(self):
        return self.x_train1, self.x_test1, self.y_train1, self.y_test1
    def get_graydata(self):
        return self.x_train0, self.x_test0, self.y_train0, self.y_test0
    def get_eigenfaces(self):
        return self.eigenfaces
    def get_nclasses(self):
        return self.n_classes
    def get_pca(self):
        return self.pca


    def __init__(self, picture_path = r"E:\Data\转移性质\GLCM\T2WI\train",
                 pictureT_path = r'E:\Data\转移性质\GLCM\T2WI\test'):
        self.picture_path = picture_path
        self.pictureT_path = pictureT_path

    #采集图像信息,装入pic_data_rgb，pic_data_gray，确定target_n, target, img_size和n_classes
    def get_eigen(self):
        t0 = time()
        print ("正在采集图像数据...")
        PICTURE_PATH = self.picture_path
        PICTURET_PATH = self.pictureT_path

        name = glob.glob(PICTURE_PATH + "\\*")
        name_t = glob.glob(PICTURET_PATH + "\\*")
        self.n_classes = len(name)#总的分类数
        for i in range(len(name)):
            pic_name = name[i].split('\\')[-1]
            self.target_n.append(pic_name)
            pic = glob.glob(name[i] + "\\*/*.jpg")
            for j in pic:
                img = Image.open(j)
                self.pic_data_rgb.append( img.resize((50,37)) )
                self.img_size.append(img.size)
                img = img.convert('L')#原来的data是[(r,g,b)],现转为灰度图像[grey]
                img = img.resize((40, 50))#将图片转化为统一的格式，统一之后操作数组的长度
                self.pic_data_gray.append( list(img.getdata()) )
                # print (list(img.getdata()))
                # print('-----------------------')
                self.target.append(i)
        for k in range(len(name_t)):
            # pic_name = name_t[k].split('\\')[-1]
            # self.target_tn.append(pic_name)
            pic = glob.glob(name_t[k] +"\\*/*.jpg")
            for l in pic:
                img = Image.open(l)
                self.pic_data_rgb_T.append(img.resize((50,37)))
                self.img_size.append(img.size)
                img = img.convert('L')
                img = img.resize((40,50))
                self.pic_data_gray_test.append(list(img.getdata()) )
                self.target_t.append(k)
        self.y_test = np.array(self.target_t)
        return "图片信息采集完毕！花费了%0.3f秒"%(time()-t0)

    #分出训练集和测试集，并进行降维处理
    def get_pca_eigen(self):
        print (self.get_eigen())
        print("------------------------------------------")
        n_samples = len(self.pic_data_gray)
        n_features = len(self.pic_data_gray[0])

        #test
        print("训练集的规模如下:")
        print("样本数: %d" % n_samples)
        print("特征数: %d" % n_features)
        print("分类数: %d" % self.n_classes)
        print("------------------------------------------")
        #-----------------------------------------------#
        #分出训练集和测试集


        # self.x_train0, self.x_test0, self.y_train0, self.y_test0 = train_test_split(
        #     self.pic_data_gray, self.target, test_size=0.25, random_state=42)
        self.x_train0, self.x_test0, self.y_train0, self.y_test0 = self.pic_data_gray, self.pic_data_gray_test, self.target, self.target_t
        self.x_train1, self.x_test1, self.y_test1, self.y_test1 = self.pic_data_rgb, self.pic_data_rgb_T, self.target,self.target_t
        # self.x_train1, self.x_test1, self.y_train1, self.y_test1 = train_test_split(
        #     self.pic_data_rgb, self.target, test_size=0.25, random_state=42)
        #-----------------------------------------------#
        #PCA降维处理

        n_components = 4#这个降维后的特征值个数如果太大，比如100，结果将极其不准确，为何？？
        print("正在进行降维处理...")
        t0 = time()
        self.pca = PCA(n_components = n_components, svd_solver='auto',
                  whiten=True).fit(self.x_train0)

        #PCA降维后的数据集
        x_train_pca = self.pca.transform(self.x_train0)
        x_test_pca = self.pca.transform(self.x_test0)
        df2 = pd.DataFrame(np.array(x_test_pca))
        df2.to_csv("pca_data/x_test_T2WI.csv")
        print("完成！花费了 %0.3f秒" % (time() - t0))
        #X为降维后的数据，y是对应类标签

        self.x_train = np.array(x_train_pca)#训练集
        self.x_test = np.array(x_test_pca)#测试集
        self.y_train = np.array(self.y_train0)#训练集target
        self.y_test = np.array(self.y_test0)#测试集target
        self.x_train0 = np.array(self.x_train0)
        self.x_test0 = np.array(self.x_test0)
        return "数据分类降维完毕！"
