import os
import time

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet
from src import fcn_resnet50
from src import deeplabv3_resnet50

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

#  18_23_M_SER_1005_1F  18_33_M_SER_1002_2D  18_13_B_SUB3_32
def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model_VibrantUnet.pth"
    img_path = r"E:\Data\Mydataset\segment\test\images\18_46_M_SER_1001_31.jpg"
    roi_mask_path = r"E:\Data\Mydataset\segment\test\3st_manual/18_13_B_VIBRANT_32.bmp"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # aux=False
    # model = fcn_resnet50(aux=aux, num_classes=classes + 1)
    # # delete weights about aux_classifier
    # weights_dict = torch.load(weights_path, map_location='cpu')['model']
    # for k in list(weights_dict.keys()):
    #     if "aux" in k:
    #         del weights_dict[k]
    #
    # # load weights
    # model.load_state_dict(weights_dict)
    # model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = cv2.imread(img_path,1)
    original_img = Image.fromarray(original_img)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("unet4.png")


if __name__ == '__main__':
    main()
