"""
    DataSet库说明-
    读取文件路径-img_dir
    读取50,000张图片
    输出为 X, Y
    X是（10000，4,56,56） 1000表示样本数 4表示通道数（包含四个emcal hcal trkn trkp)
    Y是（10000,56,56）
"""
# 导入相关库
import os # 与系统文件交互
import tifffile as tiff #读取tiff文件格式
from PIL import Image #图片处理
#与torch 相关的库
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import imageio 

class MaxMinNormalizeGlobalPerChannel:
    """
    针对 (Batch, Channel, Width, Height) 的张量，
    当mode为global,则在所有 Batch 中对每个通道整体进行最大最小归一化。
    当mode为per_channel,则在每个 Batch 中对每个通道单独进行最大最小归一化。
    """
    def __init__(self, mode='global'):
        self.mode = mode  # 'global' 或 'per_channel'

    def __call__(self, tensor):
        assert tensor.dim() == 4
        
        if self.mode == 'global':
            dims = (0,1,2,3)
        elif self.mode == 'per_channel':
            dims = (0,2,3)
            
            
        min_val = tensor.amin(dim=dims, keepdim=True)
        max_val = tensor.amax(dim=dims, keepdim=True)
        
        print(f"Min: {min_val.squeeze()}")
        print(f"Max: {max_val.squeeze()}")

        return (tensor - min_val) / (max_val - min_val + 1e-8)
    

#创建数据集
class MyDataSet(Dataset):
    def __init__(self,img_dir,group_size=10000,size_in=10000,transform=None,
                split_shuffle = True,splition = False):
        self.img_dir=img_dir
        self.images=os.listdir(img_dir)
        self.transform=transform
        self.all_imgs=[]
        self.emcal=[]
        self.hcal=[]
        self.trkn=[]
        self.trkp=[]
        self.truth=[]
        self.group_size=group_size
        self.size_in=size_in
        self.splition=splition
        self.split_shuffle = split_shuffle
        self.load_images()
        #self.normalize()
    
    def load_images(self):
        all_imgs=[]
        to_pil = transforms.ToPILImage()
        prefixes = ['emcal', 'hcal', 'trkn', 'trkp', 'truth']
        for prefix in prefixes:
            for i in range(self.size_in):
                filename = f"{prefix}_{str(i)}.tiff"
                img_path = img_path=os.path.join(self.img_dir, filename)
                # print(img_path)
                img_array=tiff.imread(img_path)
                img=Image.fromarray(img_array)
                img_tensor=transform(img)
                all_imgs.append(img_tensor)
        self.emcal=all_imgs[:self.size_in]
        self.hcal=all_imgs[self.group_size:self.group_size+self.size_in]
        self.trkn=all_imgs[2*self.group_size:2*self.group_size+self.size_in]
        self.trkp=all_imgs[3*self.group_size:3*self.group_size+self.size_in]
        self.truth=all_imgs[4*self.group_size:4*self.group_size+self.size_in]
        
        self.X=[]
        self.Y=[]
        picture = np.ndarray([])
        
        if self.transform is not None:
            transformation = self.transform
            print('transformation is not None')
        else:
            transformation = lambda x: x
            print('transformation is None')
        
        for emcal, hcal, trkn, trkp in zip(self.emcal,self.hcal,self.trkn, self.trkp):
            combined_features=torch.stack((emcal,hcal,trkn,trkp),dim=0).reshape(-1,56,56)
            self.X.append(combined_features)
        
        self.X=torch.stack(self.X).squeeze()
        self.X=transformation(self.X)
        self.Y=torch.stack(self.truth)
        self.Y=transformation(self.Y)
        
        N = self.X.size(0)
        train_size = int(0.8 * N)
        val_size = int(0.1 * N)
        if self.split_shuffle:
            indices = torch.randperm(N)

        else:
            indices = torch.arange(N)
            # 按照比例划分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        if self.splition == True:
            # 根据索引划分数据集
            self.train_X = self.X[train_indices]
            self.train_Y = self.Y[train_indices]
            self.val_X = self.X[val_indices]
            self.val_Y = self.Y[val_indices]
            self.test_X = self.X[test_indices]
            self.test_Y = self.Y[test_indices]
            # 释放内存
            del self.X
            del self.Y

    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]
transform=transforms.Compose([
    transforms.ToTensor(),
    # 数据预处理后期添加
])

    
class dataset_2(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]


