import os
import torch
import torch.utils.data
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2

import transforms


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "all_images")) if i.endswith(".jpg")]
        self.img_list = [os.path.join(data_root, "all_images", i) for i in img_names]
        mask_names = [i for i in os.listdir(os.path.join(data_root, "all_manual")) if i.endswith(".bmp")]
        self.mask_list = [os.path.join(data_root, "all_manual", i.split(".")[0]+".bmp") for i in img_names]
        # self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
        #                for i in img_names]
        # # check files
        # for i in self.manual:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")
        #
        # self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
        #                  for i in img_names]
        # check files
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        # img = Image.open(self.img_list[idx])
        # img = cv2.imread(self.img_list[idx],1)
        img = cv2.imdecode(np.fromfile(self.img_list[idx], dtype=np.uint8),-1)
        img = Image.fromarray(img)
        # manual = Image.open(self.manual[idx]).convert('L')
        # manual = np.array(manual) / 255
        roi_mask = Image.open(self.mask_list[idx]).convert('L')
        roi_mask = np.array(roi_mask) / 255
        # mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(roi_mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        # rows,cols = mask.shape
        # for i in range(rows):
        #     for j in range(cols):
        #         if mask[i,j]!=255:
        #             print(mask[i,j])
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

if __name__ == '__main__':
    val_dataset = DriveDataset(root=r'E:\Data\Mydataset\segment',train=False,transforms=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(val_dataset,batch_size=2,shuffle=True,collate_fn=val_dataset.collate_fn)
    for image,target in train_loader:
        print(target.shape)



