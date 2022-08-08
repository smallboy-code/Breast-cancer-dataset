import os
from glob import glob

import cv2
import numpy as np
from skimage import feature


def create_binary_pattern(img, p, r,i):

    print ('[INFO] Computing local binary pattern features. ',i)
    lbp = feature.local_binary_pattern(img, p, r)
    return (lbp-np.min(lbp))/(np.max(lbp)-np.min(lbp)) * 255


if __name__ == '__main__':
    image_dir = r'E:\Data\Mydataset\segment\training\images'
    filelist = glob(os.path.join(image_dir, '*.jpg'))
    # print(filelist)
    # path = 'E:\Data\Mydataset\segment/training\images/18_36_B_VIBRANT_41.jpg'
    image_list= []
    for i,file in enumerate(filelist):
        img = cv2.imread(file,1)
        image_list.append(img)
    for i, img in enumerate(image_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        create_binary_pattern(img_gray, 24 * 8, 8,i)