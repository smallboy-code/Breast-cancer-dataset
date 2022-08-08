import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
def change():
    path = r'E:\Data\Mydataset\segment\test\mask'
    for mask in os.listdir(path):
        mask_path = os.path.join(path,mask)
        a = Image.open(mask_path).convert('L')
        img = np.array(a)
        rows,cols = img.shape
        # for i in range(rows):
        #     for j in range(cols):
        #         if img[i,j]!=255:
        #            print(img[i,j])
        for i in range(rows):
            for j in range(cols):
                if img[i,j]!=255:
                    img[i,j]=0
    # plt.figure('lena')
    # plt.imshow(img,cmap='gray')
    # plt.axis('off')
    # plt.savefig(r'E:\Data\Mydataset\segment\training\1st_manual\1.bmp')
    # plt.show()
        cv2.imwrite(f'E:\Data\Mydataset\segment/test/1st_manual/{mask.split(".")[0]}.bmp',img)
    print('done')

def lookPth():
    path = './save_weights/best_model_AllDeepLabv3.pth'
    model = torch.load(path)
    print(model['epoch'])


if __name__ == '__main__':
    lookPth()
