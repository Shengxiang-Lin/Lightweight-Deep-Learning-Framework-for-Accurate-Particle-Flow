import numpy as np
import random 
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold
from DataSet_2 import MaxMinNormalizeGlobalPerChannel,MyDataSet
from train_and_eval import train_one_epoch, evaluate, MixedMSE

''''''''''''''''
每一次训练之前把对应的模型导入和model_dict去掉注释就行
'''''''''''''''

from model_1 import CNN_1, CNNwithSEBlock_1, CNN3D_1, UNet_1, UNetwithSEBlock_1, UNetwithSelfattention_1, UNet3D_1
# from model_2 import CNN_2, CNNwithSEBlock_2, CNN3D_2, UNet_2, UNetwithSEBlock_2, UNetwithSelfattention_2, UNet3D_2
# from model_3 import CNN_3, CNNwithSEBlock_3, CNN3D_3, UNet_3, UNetwithSEBlock_3, UNetwithSelfattention_3, UNet3D_3
# from model_4 import CNN_4, CNNwithSEBlock_4, CNN3D_4, UNet_4, UNetwithSEBlock_4, UNetwithSelfattention_4, UNet3D_4
# from model_5 import UNet_5, UNetwithSEBlock_5, UNetwithSelfattention_5, UNet3D_5
# from model_6 import UNet_6, UNetwithSEBlock_6, UNetwithSelfattention_6, UNet3D_6
# from model_final import CNNwithSEBlock3D_1, CNNwithSEBlock3D_2, CNNwithSEBlock3D_3, CNNwithSEBlock3D_4, UNetwithSEBlock3D_1, UNetwithSEBlock3D_2, UNetwithSEBlock3D_3, UNetwithSEBlock3D_4, UNetwithSEBlock3D_5, UNetwithSEBlock3D_6, UNetwithSelfattention3D_1, UNetwithSelfattention3D_2, UNetwithSelfattention3D_3, UNetwithSelfattention3D_4, UNetwithSelfattention3D_5, UNetwithSelfattention3D_6

random.seed(26)
np.random.seed(26)
torch.manual_seed(26)
torch.cuda.manual_seed(26)
torch.cuda.manual_seed_all(26) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或者 ":4096:8"

model_dict = {
    'CNN_1': CNN_1,
    'CNNwithSEBlock_1': CNNwithSEBlock_1,
    'CNN3D_1': CNN3D_1,
    'UNet_1': UNet_1,
    'UNetwithSEBlock_1': UNetwithSEBlock_1,
    'UNetwithSelfattention_1': UNetwithSelfattention_1,
    'UNet3D_1': UNet3D_1
}

# model_dict = {
#     'CNN_2': CNN_2,
#     'CNNwithSEBlock_2': CNNwithSEBlock_2,
#     'CNN3D_2': CNN3D_2,
#     'UNet_2': UNet_2,
#     'UNetwithSEBlock_2': UNetwithSEBlock_2,
#     'UNetwithSelfattention_2': UNetwithSelfattention_2,
#     'UNet3D_2': UNet3D_2
# }

# model_dict = {
#     'CNN_3': CNN_3,
#     'CNNwithSEBlock_3': CNNwithSEBlock_3,
#     'CNN3D_3': CNN3D_3,
#     'UNet_3': UNet_3,
#     'UNetwithSEBlock_3': UNetwithSEBlock_3,
#     'UNetwithSelfattention_3': UNetwithSelfattention_3,
#     'UNet3D_3': UNet3D_3
# }

# model_dict = {
#     'CNN_4': CNN_4,
#     'CNNwithSEBlock_4': CNNwithSEBlock_4,
#     'CNN3D_4': CNN3D_4,
#     'UNet_4': UNet_4,
#     'UNetwithSEBlock_4': UNetwithSEBlock_4,
#     'UNetwithSelfattention_4': UNetwithSelfattention_4,
#     'UNet3D_4': UNet3D_4
# }

# model_dict = {
#     'UNet_5': UNet_5,
#     'UNetwithSEBlock_5': UNetwithSEBlock_5,
#     'UNetwithSelfattention_5': UNetwithSelfattention_5,
#     'UNet3D_5': UNet3D_5
# }

# model_dict = {
#     'UNet_6': UNet_6,
#     'UNetwithSEBlock_6': UNetwithSEBlock_6,
#     'UNetwithSelfattention_6': UNetwithSelfattention_6,
#     'UNet3D_6': UNet3D_6
# }

# model_dict = {
#     'CNNwithSEBlock3D_1': CNNwithSEBlock3D_1,
#     'CNNwithSEBlock3D_2': CNNwithSEBlock3D_2,
#     'CNNwithSEBlock3D_3': CNNwithSEBlock3D_3,
#     'CNNwithSEBlock3D_4': CNNwithSEBlock3D_4,
#     'UNetwithSEBlock3D_1': UNetwithSEBlock3D_1,
#     'UNetwithSEBlock3D_2': UNetwithSEBlock3D_2,
#     'UNetwithSEBlock3D_3': UNetwithSEBlock3D_3,
#     'UNetwithSEBlock3D_4': UNetwithSEBlock3D_4,
#     'UNetwithSEBlock3D_5': UNetwithSEBlock3D_5,
#     'UNetwithSEBlock3D_6': UNetwithSEBlock3D_6,
#     'UNetwithSelfattention3D_1': UNetwithSelfattention3D_1,
#     'UNetwithSelfattention3D_2': UNetwithSelfattention3D_2,
#     'UNetwithSelfattention3D_3': UNetwithSelfattention3D_3,
#     'UNetwithSelfattention3D_4': UNetwithSelfattention3D_4,
#     'UNetwithSelfattention3D_5': UNetwithSelfattention3D_5,
#     'UNetwithSelfattention3D_6': UNetwithSelfattention3D_6
# }

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        :param patience: 如果在多少个epoch内验证集损失没有改善，则提前停止训练
        :param delta: 在认为损失有改善时，损失变化的最小值
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement in validation loss for {self.patience} epochs).")
                self.early_stop = True

# 在每次训练之前根据模型名实例化模型
def get_model(model_name):
    return model_dict[model_name]()

def kfold_train_and_evaluate(model_name, data_set, k, epochs, batch_size, device, earlystoplimit, lr):
    # 使用KFold进行数据划分
    kfold = KFold(n_splits=k, shuffle=True, random_state=26)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_set)):
        print(f"Training fold {fold + 1}/{k} for model {model_name}")

        # 划分训练集和验证集
        train_subset = torch.utils.data.Subset(data_set, train_idx)
        val_subset = torch.utils.data.Subset(data_set, val_idx)

        # 创建DataLoader
        train_loader = torch.utils.data.DataLoader(train_subset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_subset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0)

        # 获取模型
        model = get_model(model_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        loss_function = MixedMSE(1, 0.05)

        early_stopping = EarlyStopping(patience=10, delta=earlystoplimit)
        
        # 记录训练时间
        start_time = time.time()
        # 训练和验证
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, loss_function)
            scheduler.step()
            val_loss = evaluate(model, val_loader, device, loss_function)
            
            # 输出每个epoch的损失
            print(f"Fold {fold + 1} - Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                break
        # 保存每折结果
        # torch.save(model.state_dict(), f"{model_name}_fold{fold + 1}.pth") #暂时不保存模型
        training_time = time.time() - start_time
        print(f"Fold {fold + 1} training completed in {training_time:.2f} seconds.")
        fold_results.append({'train_loss': train_loss, 'val_loss': val_loss, 'training_time': training_time})

    # 计算并返回每折的均值和方差
    train_losses = [result['train_loss'] for result in fold_results]
    val_losses = [result['val_loss'] for result in fold_results]
    training_times = [result['training_time'] for result in fold_results]

    return {
        'model_name': model_name,
        'train_loss_mean': np.mean(train_losses),
        'train_loss_std': np.std(train_losses),
        'val_loss_mean': np.mean(val_losses),
        'val_loss_std': np.std(val_losses),
        'avg_training_time': np.mean(training_times),
    }

def main(args):

    data_transform = {
        "without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()]),
        "jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()])}

    # 实例化训练数据集
    data_set = MyDataSet(img_dir=args.img_dir,
                        group_size=10000,
                        size_in = 10000,
                        splition = False,
                        split_shuffle = False,
                        transform=data_transform["without_jet"])

    # 存储所有模型的结果
    all_results = []


    # 训练每个模型并记录结果
    for model_name in model_dict.keys():
        result = kfold_train_and_evaluate(model_name, data_set, k=args.k, epochs=args.epochs, batch_size=args.batch_size,
                                        device=args.device, earlystoplimit=args.earlystoplimit, lr=args.lr)
        all_results.append(result)

    # 输出所有模型的结果
    for result in all_results:
        print(f"Model: {result['model_name']}")
        print(f"Train Loss Mean: {result['train_loss_mean']}, Train Loss Std: {result['train_loss_std']}")
        print(f"Validation Loss Mean: {result['val_loss_mean']}, Validation Loss Std: {result['val_loss_std']}")
        print(f"Average Training Time: {result['avg_training_time']}")
        print("-" * 50)

class Args:
    def __init__(self):
        self.epochs = 500
        self.batch_size = 200
        self.lr = 0.001
        self.img_dir = 'Gauss_S1.00_NL0.30_B0.50\Gauss_S1.00_NL0.30_B0.50' 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.earlystoplimit = 1e-6
        self.k = 5

opt = Args()

if __name__ == "__main__":
    main(opt)