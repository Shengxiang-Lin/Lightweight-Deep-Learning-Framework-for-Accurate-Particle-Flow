import torch
from torch.utils.data import Dataset, DataLoader,random_split
class dataset_2(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]
    