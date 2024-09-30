import tifffile as tiff 
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class myDataSet(Dataset):
    def __init__(self,img_dir,transform=None):
        # 注意这里的文件路径 img_dir
        self.img_dir=img_dir
        self.images=os.listdir(img_dir)
        self.transform=transform
        self.all_imgs=[]
        self.emcal=[]
        self.hcal=[]
        self.tracker=[]
        self.truth=[]
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

        p=1000

        self.emcal=all_imgs[:p]
        self.hcal=all_imgs[p:2*p]
        self.tracker=all_imgs[2*p:3*p]
        self.truth=all_imgs[3*p:4*p]

        self.X=[]
        self.Y=[]
        
        for emcal, hcal, tracker in zip(self.emcal,self.hcal,self.tracker):
            combined_features=torch.stack((emcal,hcal,tracker))
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
    transforms.Normalize(mean=0,std=3.6)
])


