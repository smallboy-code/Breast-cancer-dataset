import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import pickle as pkl
import train
import math

def check_args(args):

    if not os.path.exists(args.image_dir):
        raise ValueError("Image directory does not exist")

    if not os.path.exists(args.output_dir):
        raise ValueError("Output directory does not exist")

    return args
# 18_33_M_SER_1002_2D  18_46_M_SER_1001_31  18_13_B_SUB3_32
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir" , help="Path to images", default=r'E:\Data\Mydataset\segment\test\sub_images\18_13_B_SUB3_32.jpg')
    parser.add_argument("-m", "--model_path", help="Path to .p model", default='./model/RFmodel_SUB.p')
    parser.add_argument("-o", "--output_dir", help="Path to output directory", default='./pots/outputs')
    args = parser.parse_args()
    return check_args(args)

def create_features(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features, _ = train.create_features(img, img_gray, label=None, train=False)

    return features

def compute_prediction(img, model):

    # border = 5 # (haralick neighbourhood - 1) / 2
    #
    # img = cv2.copyMakeBorder(img, top=border, bottom=border, \
    #                               left=border, right=border, \
    #                               borderType = cv2.BORDER_CONSTANT, \
    #                               value=[0, 0, 0])

    features = create_features(img)
    predictions = model.predict(features.reshape(-1, features.shape[1]))
    # pred_size = int(math.sqrt(features.shape[0]))

    inference_img = predictions.reshape(img.shape[0], img.shape[1])

    return inference_img

def infer_images(image_dir, model_path, output_dir):

    # filelist = glob(os.path.join(image_dir,'*.jpg'))

    # print ('[INFO] Running inference on %s test images' %len(filelist))

    model = pkl.load(open( model_path, "rb" ) )

    # for file in filelist:
    print ('[INFO] Processing images:', os.path.basename(image_dir))
    inference_img = compute_prediction(cv2.imread(image_dir, 1), model)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_dir)), inference_img)

def main(image_dir, model_path, output_dir):

    infer_images(image_dir, model_path, output_dir)

if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    model_path = args.model_path
    output_dir = args.output_dir
    main(image_dir, model_path, output_dir)
