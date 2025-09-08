import os
import random
import numpy as np
import time

import torch
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from DataSet_2 import MaxMinNormalizeGlobalPerChannel,MyDataSet, dataset_2
from train_and_eval import train_one_epoch, evaluate, MixedMSE
from CNN import CNN, CNNwithSEBlock, CNNwithRowSelfAttention
from MB_model import MB_ThreeBranch, MB_ThreeBranch_v2
from deepset import DSReconstruction
from GNN import DGCNNModel

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
    'cnn': CNN,
    'cnn_se': CNNwithSEBlock,
    'cnn_row_attn': CNNwithRowSelfAttention,
    'deepset': DSReconstruction,
    'gnn': DGCNNModel,
    'MB_ThreeBranch': MB_ThreeBranch,
    'MB_ThreeBranch_v2': MB_ThreeBranch_v2
}


class EarlyStopping:
    def __init__(self, patience=20, delta=0):
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
    # 确保output目录存在
    import os
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为每个模型创建训练日志文件，保存到output目录
    log_file_path = os.path.join(output_dir, f"{model_name}_training_log.txt")
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"=== {model_name} 训练开始 ===\n")
        log_file.write(f"训练参数: epochs={epochs}, lr={lr}, earlystoplimit={earlystoplimit}\n")
        log_file.write(f"设备: {device}\n")
        log_file.write("=" * 50 + "\n")
    
    model = get_model(model_name).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_function = MixedMSE(1, 0.05)
    early_stopping = EarlyStopping(patience=20, delta=earlystoplimit)
    start_time = time.time()

    best_model = model
    best_val_loss = 10000
    
    # 记录训练过程
    with open(log_file_path, 'a') as log_file:
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, optimizer, testloader, device, epoch, loss_function)
            scheduler.step()
            val_loss = evaluate(model, valloader, device, loss_function)
            
            # 输出每个epoch的损失
            epoch_info = f" Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}"
            print(epoch_info)
            log_file.write(epoch_info + "\n")
            log_file.flush()  # 立即写入文件

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if epoch > 10 :#设置模型保存间隔
                    best_model = model
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                stop_info = f"Early stopping at epoch {epoch + 1} (no improvement in validation loss for {early_stopping.patience} epochs)."
                print(stop_info)
                log_file.write(stop_info + "\n")
                break
    
    # 保存模型到output目录
    model_save_path = os.path.join(output_dir, f"{model_name}.pth")
    torch.save(best_model.state_dict(), model_save_path)
    training_time = time.time() - start_time
    
    # 记录最终结果
    final_result = {
        'model_name': model_name,
        'model_loss': best_val_loss,
        'training_time': training_time,
    }
    
    # 立即保存结果到独立文件，保存到output目录
    result_file_path = os.path.join(output_dir, f"{model_name}_result.txt")
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Model: {model_name}\n")
        result_file.write(f"Validation Loss: {best_val_loss}\n")
        result_file.write(f"Training Time: {training_time}\n")
        result_file.write("-" * 50 + "\n")
    
    # 同时追加到总结果文件，也保存到output目录
    results_summary_path = os.path.join(output_dir, "results.txt")
    with open(results_summary_path, 'a') as file:
        output = (
            f"Model: {model_name}\n"
            f"Validation Loss: {best_val_loss}\n"
            f"Training Time: {training_time}\n"
            f"{'-' * 50}\n"
        )
        file.write(output)
    
    # 记录训练完成信息到日志文件
    with open(log_file_path, 'a') as log_file:
        log_file.write("=" * 50 + "\n")
        log_file.write(f"训练完成！\n")
        log_file.write(f"最佳验证损失: {best_val_loss}\n")
        log_file.write(f"总训练时间: {training_time:.2f} 秒\n")
        log_file.write(f"模型已保存为: {model_save_path}\n")
        log_file.write("=" * 50 + "\n")
    
    return final_result


def main(args):

    data_transform = {
        "without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel(mode='global')]),
        "jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel(mode='global')]),}
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
    print(len(train_dataset))
    print(len(test_dataset))
    
    all_results = []
    # 为v2和v3版本模型设置更小的batch size
    model_batch_sizes = {
        'CNN':128,
        'CNNwithSEBlock': 128,
        'CNNwithRowSelfAttention': 80,
        'deepset': 256,
        'gnn': 8,
        'MB_ThreeBranch': 200,
        'MB_ThreeBranch_v2': 128,
    }
    
    # 训练每个模型并记录结果
    for model_name in model_dict.keys():
        print(f"\n{'='*60}")
        print(f"开始训练模型: {model_name}")
        print(f"{'='*60}")
        
        # 根据模型名称选择batch size
        batch_size = model_batch_sizes.get(model_name, 200)
        print(f"使用batch size: {batch_size}")

        # 基于batch size计算学习率：当batch=200时lr=0.001，否则按比例缩放
        base_lr = 0.001
        lr_for_model = base_lr * (batch_size / 200.0)
        print(f"使用学习率: {lr_for_model}")

        # 为当前模型创建DataLoader
        current_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        current_val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        result = train(model_name, current_train_dataloader, current_val_dataloader, epochs=args.epochs,
                                        device=args.device, earlystoplimit=args.earlystoplimit, lr=lr_for_model)
        all_results.append(result)
        
        # 立即输出当前模型的结果
        print(f"\n模型 {model_name} 训练完成:")
        print(f"Validation Loss: {result['model_loss']}")
        print(f"Training Time: {result['training_time']}")
        print("-" * 50)

    # 输出所有模型的结果总结
    print(f"\n{'='*60}")
    print("所有模型训练完成！结果总结:")
    print(f"{'='*60}")
    for result in all_results:
        print(f"Model: {result['model_name']}")
        print(f"Validation Loss: {result['model_loss']}")
        print(f"Training Time: {result['training_time']}")
        print("-" * 50)

class Args:
    def __init__(self):
        self.epochs = 100000
        self.batch_size = 200
        self.lr = 0.001
        self.img_dir = 'D:\\LECINSUMMER\\project4\\Gauss_S1.00_NL0.30_B0.50\\Gauss_S1.00_NL0.30_B0.50' 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.earlystoplimit = 0


opt = Args()

if __name__ == "__main__":
    main(opt)
