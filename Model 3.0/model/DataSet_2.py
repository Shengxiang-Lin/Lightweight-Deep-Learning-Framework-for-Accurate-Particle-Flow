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
    def __init__(self, mode='global', max_val=None, min_val=None):
        self.mode = mode  # 'global' 或 'per_channel'
        self.max_val = max_val
        self.min_val = min_val

    def __call__(self, tensor):
        assert tensor.dim() == 4
        
        if self.max_val is not None and self.min_val is not None:
            max_val = self.max_val.reshape(1, -1, 1, 1)
            min_val = self.min_val.reshape(1, -1, 1, 1)
        else:
            if self.mode == 'global':
                dims = (0,1,2,3)
            elif self.mode == 'per_channel':
                dims = (0,2,3)
            self.min_val = tensor.amin(dim=dims, keepdim=True)
            self.max_val = tensor.amax(dim=dims, keepdim=True)
            min_val, max_val = self.min_val, self.max_val
        print(f"Min: {min_val.squeeze()}")
        print(f"Max: {max_val.squeeze()}")

        return (tensor - min_val) / (max_val - min_val + 1e-8), min_val, max_val
    

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
        to_pil = transforms.ToPILImage()
        
        # 分别为每种类型创建列表
        self.emcal = []
        self.hcal = []
        self.trkn = []
        self.trkp = []
        self.truth = []
        
        # 按照文件名编号顺序读取每种类型的图片
        for i in range(self.size_in):
            # 读取emcal图片
            emcal_filename = f"emcal_{i}.tiff"
            emcal_path = os.path.join(self.img_dir, emcal_filename)
            emcal_array = tiff.imread(emcal_path)
            emcal_img = Image.fromarray(emcal_array)
            emcal_tensor = transform(emcal_img)
            self.emcal.append(emcal_tensor)
            
            # 读取hcal图片
            hcal_filename = f"hcal_{i}.tiff"
            hcal_path = os.path.join(self.img_dir, hcal_filename)
            hcal_array = tiff.imread(hcal_path)
            hcal_img = Image.fromarray(hcal_array)
            hcal_tensor = transform(hcal_img)
            self.hcal.append(hcal_tensor)
            
            # 读取trkn图片
            trkn_filename = f"trkn_{i}.tiff"
            trkn_path = os.path.join(self.img_dir, trkn_filename)
            trkn_array = tiff.imread(trkn_path)
            trkn_img = Image.fromarray(trkn_array)
            trkn_tensor = transform(trkn_img)
            self.trkn.append(trkn_tensor)
            
            # 读取trkp图片
            trkp_filename = f"trkp_{i}.tiff"
            trkp_path = os.path.join(self.img_dir, trkp_filename)
            trkp_array = tiff.imread(trkp_path)
            trkp_img = Image.fromarray(trkp_array)
            trkp_tensor = transform(trkp_img)
            self.trkp.append(trkp_tensor)
            
            # 读取truth图片
            truth_filename = f"truth_{i}.tiff"
            truth_path = os.path.join(self.img_dir, truth_filename)
            truth_array = tiff.imread(truth_path)
            truth_img = Image.fromarray(truth_array)
            truth_tensor = transform(truth_img)
            self.truth.append(truth_tensor)
        
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
        self.Y=torch.stack(self.truth)

        self.X, min_vals, max_vals =transformation(self.X)
        global_min = torch.min(min_vals) if min_vals.numel() == 1 else torch.sum(min_vals)
        global_min = global_min.view(1, -1, 1, 1)
        print(f"Global Min: {global_min}")

        global_max = torch.max(max_vals) if max_vals.numel() == 1 else torch.sum(max_vals)
        global_max = global_max.view(1, -1, 1, 1)
        print(f"Global Max: {global_max}")
        # 对truth使用四层的全局极值
        print(f"Y max: {torch.max(self.Y)}")
        self.Y = (self.Y - global_min) / (global_max - global_min + 1e-8)
        
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


