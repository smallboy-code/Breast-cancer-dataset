# coding: utf-8
import glob
import os.path

import numpy as np
from skimage import data
from matplotlib import pyplot as plt
#from fast_glcm import fast_glcm_mean
import fast_glcm
import cv2

def main():
    pass


if __name__ == '__main__':
    main()
    path = r'E:\Data\转移性质\T2WI\1'
    img_path = glob.glob(os.path.join(path,'*/*.jpg'))
    for imgs in img_path:
        img = cv2.imdecode(np.fromfile(imgs, dtype=np.uint8),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h,w = img.shape
        glcm_mean = fast_glcm.fast_glcm_contrast(img)
        new_path = os.path.join(r'E:\Data\转移性质\GLCM/T2WI/1',imgs.split('\\')[-2])
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        plt.imsave(os.path.join(new_path,imgs.split('\\')[-1]),glcm_mean)

