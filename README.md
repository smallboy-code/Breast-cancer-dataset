# Breast-cancer-dataset
This is the codes for paper "BreastDM: A DCE-MRI Dataset for Breast Tumor Image Classification and Segmentation".  Due to patient privacy and copyright issues with the original dataset, you can apply for the dataset by sending an email to (tzczsq@163.com). 
## Classification task
### 1. Environment
see the requirements.txt
### 2. Data preparation
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
### 3. Training Demo
#### 3.1 Texture features extract models
There are three traditional texture feature extract methods in our experiments: local binary patterns(LBP), local phase quantization(LPQ) and gray-level co-occurrence matrix
(GLCM). You can run these methods directly and we provide the extracted feature sets for testing.

#### 3.2 CNN models
We use the VGG16,ResNet50,ResNet101,SENet50 models in our experiments. You can use the following command to run them.
```
python train.py --batch-size 32 --model resnet50 --gpu 0 --num_class 2 --split_train_ratio 0.8 --task_name "your task-name" --path "data dir" --auto_split 0
```

#### 3.3 Vision Transformer
You can use the following command to run vision transformer.
```
python train.py --num_classes 2 --epochs 20 --batch-size 32 --lr 0.01 --lrf 0.01 --data-path "data dir" --model-name "create model name" --weights "initial weights dir"
```
#### 3.4 Local-global cross attention network

## Segmentation task
### 


      
