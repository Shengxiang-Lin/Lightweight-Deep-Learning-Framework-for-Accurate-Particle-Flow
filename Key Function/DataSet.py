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

#创建数据集
class myDataSet(Dataset):
    def __init__(self,img_dir,group_size=10000,transform=None):
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
        self.load_images()
    
    def load_images(self):
        all_imgs=[]
        for filename in self.images:
            if filename.endswith(".tiff"):
                img_path=os.path.join(self.img_dir, filename)
            img_array=tiff.imread(img_path)
            img=Image.fromarray(img_array)
            img_tensor=transform(img)
            all_imgs.append(img_tensor)
        self.emcal=all_imgs[:self.group_size]
        self.hcal=all_imgs[self.group_size:2*self.group_size]
        self.trkn=all_imgs[2*self.group_size:3*self.group_size]
        self.trkp=all_imgs[3*self.group_size:4*self.group_size]
        self.truth=all_imgs[4*self.group_size:5*self.group_size]
        
        self.X=[]
        self.Y=[]
        
        for emcal, hcal, trkn, trkp in zip(self.emcal,self.hcal,self.trkn, self.trkp):
            combined_features=torch.stack((emcal,hcal,trkn,trkp))
            self.X.append(combined_features)
        self.X=torch.stack(self.X).squeeze()
        self.Y=torch.stack(self.truth).squeeze()
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]
transform=transforms.Compose([
    transforms.ToTensor(),
    # 数据预处理后期添加
])
    



