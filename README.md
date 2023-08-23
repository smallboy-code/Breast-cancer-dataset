# Breast-cancer-dataset
This is the codes for paper "BreastDM: A DCE-MRI Dataset for Breast Tumor Image Segmentation and Classification, Xiaoming Zhao; Yuehui Liao; Jiahao Xie; Xiaxia He; Shiqing Zhang*; Guoyu Wang*; Jiangxiong Fang; Hongsheng Lu; Jun Yu. Computers in Biology and Medicine, Doi:10.1016/j.compbiomed.2023.107255, 2023."[https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205). We have presented our dataset by the url "[drive.google.com].(https://drive.google.com/file/d/1GvNwL4iPcB2GRdK2n353bKiKV_Vnx7Qg/view?usp=drive_link).
## Classification task
### 1. Environment
see the requirements.txt
### 2. Local-global cross attention network
![image](Classification%20task/asserts/jpg/1.png)
we design two different branches: a local CNN subnetwork (i.e., SENet) for capturing local information, and a global Transformer subnetwork (i.e., ViT) for capturing global information. For the extracted feature maps from these two branches, their feature dimensions are usually different. In this case, they cannot be directly fed into down-stream cross attention fusion blocks. To address this issue, we adopt a feature coupling unit (FCU) to eliminate feature dimension mis-alignment between these two branches. For cross attention fusion, we employ a non-local operation block used in non-local neural networks as a basic block, due to its good ability of capturing long-range dependencies.
### 3. Data preparation
if you want to use the codes for your own data,you need to put the data in the "data" directory in the following structure.
```
  --dataset
    --B
      -- img1.jpg
      -- img2.jpg
    --M
      -- img3.jpg
      -- img4.jpg
```
### 4. Training
#### 4.1 Texture features extract models
There are three traditional texture feature extract methods in our experiments: local binary patterns(LBP), local phase quantization(LPQ) and gray-level co-occurrence matrix
(GLCM). You can run these methods directly and we provide the extracted feature sets for testing.

#### 4.2 CNN models
We use the VGG16,ResNet50,ResNet101,SENet50 and our LG-CAN models in our experiments. You can use the following command to run them.
```
python train.py --batch-size 32 --model "model_name" --gpu 0 --num_class 2 --split_train_ratio 0.8 --task_name "your task-name" --path "data dir" --auto_split 0
```
#### 4.3 Vision Transformer
You can use the following command to run vision transformer.
```
python train.py --num_classes 2 --epochs 20 --batch-size 32 --lr 0.01 --lrf 0.01 --data-path "data dir" --model-name "create model name" --weights "initial weights dir"
```
### 5 Test
```
python mytest.py
```

## Segmentation task

### 1. Data Format
The files should be putting as the following structure.
```
files
└── <dataset>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 001.png
        ├── 002.png
        ├── 003.png
        ├── ...
        
        
```

### 2. Training
You can use thr following command to run FCN,Unet,DeepLabv3.
```
python train.py --data-path "data dir" --num-classes 1 --batch-size 64 --epochs 100 --lr 0.01 
```
Train the UNeXt
```
python train.py --dataset <dataset name> --arch UneXt --name <exp name> --img_ext .png --mask_ext .png --lr 0.0001 --epochs 500 --input_w 512 --input_h 512 --b 8
```
### 3. Test

```
python val.py
```

      
