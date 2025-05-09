import os
import random
import numpy as np
import time

import torch
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model_train import CNN, CNNwithSEBlock, CNN3D, CNNwithSEBlock3D, UNet, UNetwithSEBlock, UNetwithSelfattention, UNet3D, UNetwithSEBlock3D, UNetwithSelfattention3D

from DataSet_2 import MaxMinNormalizeGlobalPerChannel,MyDataSet, dataset_2
from train_and_eval import train_one_epoch, evaluate

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
    'CNN': CNN,
    'CNNwithSEBlock': CNNwithSEBlock,
    'CNN3D': CNN3D,
    'CNNwithSEBlock3D': CNNwithSEBlock3D,
    # 'UNet': UNet,
    # 'UNetwithSEBlock': UNetwithSEBlock,
    # 'UNetwithSelfattention': UNetwithSelfattention,
    # 'UNet3D': UNet3D,
    # 'UNetwithSEBlock3D': UNetwithSEBlock3D,
    # 'UNetwithSelfattention3D': UNetwithSelfattention3D,
}


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

def train(model_name, testloader, valloader, epochs, device, earlystoplimit, lr):
    model = get_model(model_name).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_function = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=20, delta=earlystoplimit)
    start_time = time.time()

    best_model = model
    best_val_loss = 10000
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, testloader, device, epoch, loss_function)
        scheduler.step()
        val_loss = evaluate(model, valloader, device, loss_function)
        
        # 输出每个epoch的损失
        print(f" Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if epoch > 50 :#设置模型保存间隔
                best_model = model
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            break
    torch.save(best_model.state_dict(), f"{model_name}.pth")
    training_time = time.time() - start_time
    return {
        'model_name': model_name,
        'model_loss': best_val_loss,
        'training_time': training_time,
    }


def main(args):

    data_transform = {
        "without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()]),
        "jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()])}
    # 实例化训练数据集
    data_set = MyDataSet(img_dir=args.img_dir,
                        group_size=10000,
                        size_in = 10000,
                        splition = True,
                        split_shuffle = False,
                        transform=data_transform['without_jet'])
    train_dataset = dataset_2(data_set.train_X, data_set.train_Y)
    val_dataset = dataset_2(data_set.val_X, data_set.val_Y)
    test_dataset = dataset_2(data_set.test_X, data_set.test_Y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=200, shuffle=False)
    print(len(train_dataset))
    print(len(test_dataset))
    
    all_results = []
    # 训练每个模型并记录结果
    for model_name in model_dict.keys():
        result = train(model_name, train_dataloader, val_dataloader, epochs=args.epochs,
                                        device=args.device, earlystoplimit=args.earlystoplimit, lr=args.lr)
        all_results.append(result)

    # 输出所有模型的结果
    for result in all_results:
        print(f"Model: {result['model_name']}")
        print(f"Validation Loss: {result['model_loss']}")
        print(f"Training Time: {result['training_time']}")
        print("-" * 50)

class Args:
    def __init__(self):
        self.epochs = 1000
        self.batch_size = 200
        self.lr = 0.001
        self.img_dir = 'Gauss_S1.00_NL0.30_B0.50\Gauss_S1.00_NL0.30_B0.50' 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.earlystoplimit = 0


opt = Args()

if __name__ == "__main__":
    main(opt)
