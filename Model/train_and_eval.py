import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import os


class WeightedMSELoss(nn.Module):
    """
    自定义加权MSE损失 (Weighted Mean Squared Error).
    下面以“基于真实值”来计算权重为例：
       weight = 1 + alpha * y_true
    您可自由替换成“基于预测值”或其他形式。
    """
    def __init__(self, alpha, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # weight = 1 + alpha * y_true
        # 如果 y_true 是归一化到 [0, 1]，此时 y_true 越大 -> 权重越高
        weight = 1.0 + self.alpha * y_true

        mse_element = (y_pred - y_true) ** 2
        weighted_mse = weight * mse_element

        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse  # 返回 element-wise 的张量


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    loss_function_compare = nn.MSELoss()
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, (batch_X, batch_Y) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs=model(batch_X.to(device))
        loss_train=loss_function(outputs,batch_Y.to(device))
        loss_compare = loss_function_compare(outputs, batch_Y.to(device))
        loss_train.backward()
        optimizer.step()
        
        mean_loss = (mean_loss * step + loss_compare.detach()) / (step + 1)  # update mean losses
        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 7))
        
        if not torch.isfinite(loss_train):
            print('WARNING: non-finite loss, ending training ', loss_train)
            sys.exit(1)
        
        
    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, loss_function):
    mean_loss = torch.zeros(1).to(device)
    model.eval()
    val_loss = 0
    for batch_X, batch_Y in data_loader:
        outputs=model(batch_X.to(device))
        mean_loss += loss_function(outputs,batch_Y.to(device)).detach()
    mean_loss /= len(data_loader)
    return mean_loss.item()

@torch.no_grad()
def plot_image(net, data_loader, device, label):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 5)
    net.eval()
    fig_list = []
    for batch_X, batch_Y in data_loader:
        outputs=net(batch_X.to(device)).detach()
        for i in range(plot_num):
            fig = plt.figure()
            fig.suptitle(label, fontsize=16)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(batch_Y[i].cpu().numpy().squeeze())
            ax1.axis('off')
            ax1.set_title('Ground Truth')
            ax2.imshow(outputs[i].cpu().numpy().squeeze())
            ax2.axis('off')
            ax2.set_title(f'Prediction')
            
            fig_list.append(fig)
        break
    return fig_list