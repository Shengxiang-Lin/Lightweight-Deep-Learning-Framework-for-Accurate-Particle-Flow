import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


def gaussian_kernel(window_size, sigma):
    """ 生成高斯滤波核，用于计算 SSIM """
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    return (gauss / gauss.sum()).view(1, 1, -1)

def ssim_loss(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """ 计算 SSIM 损失 """
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(1)  # (batch, 1, H, W)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(1)

    # 生成高斯核
    window = gaussian_kernel(window_size, sigma).to(img1.device)
    window = window.unsqueeze(1)  # (1, 1, window_size, window_size)

    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=1) - mu1**2
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=1) - mu2**2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1 * mu2

    # 计算 SSIM
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = ssim_map.mean()

    return 1 - ssim_score  # 让 SSIM 成为一个损失项

class MixedMSE(nn.Module):
    def __init__(self, lambda_wmse, lambda_ssim):
        """
        结合加权均方误差 (Weighted MSE) 和 结构相似性损失 (SSIM) 的损失函数
        :param lambda_wmse: 加权均方误差损失的权重
        :param lambda_ssim: 结构相似性损失（SSIM）的权重
        """
        super(MixedMSE, self).__init__()
        self.lambda_wmse = lambda_wmse
        self.lambda_ssim = lambda_ssim

    def forward(self, y_pred, y_true):
        """
        计算损失
        :param y_pred: 预测值 (batch_size, H, W)
        :param y_true: 真实值 (batch_size, H, W)
        :return: 计算得到的混合损失
        """

        # 加权均方误差 (Weighted MSE Loss)
        squared_diff = (y_pred - y_true) ** 2  # (batch_size, H, W)
        weight = y_true  # 真实值作为权重 (batch_size, H, W)
        weighted_mse = torch.sum(weight * squared_diff, dim=(1, 2)) / (torch.sum(y_true, dim=(1, 2)) + 1e-8)  # 避免除零
        weighted_mse_loss = torch.mean(weighted_mse)  # 取 batch 维度的均值

        # 结构相似性损失 (SSIM Loss)
        ssim_loss_value = ssim_loss(y_pred, y_true)

        # 组合损失
        total_loss = self.lambda_wmse * weighted_mse_loss + self.lambda_ssim * ssim_loss_value

        return total_loss




class WeightedMSELoss(nn.Module):
    def __init__(self):
        """
        自定义损失函数
        :param lambda1: 能量守恒损失的权重
        :param lambda2: 加权均方误差损失的权重
        """
        super(WeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        计算损失
        :param y_pred: 预测值 (batch_size, H, W)
        :param y_true: 真实值 (batch_size, H, W)
        :return: 计算得到的加权损失
        """
        batch_size = y_pred.shape[0]

        # 计算能量守恒损失 (Energy Conservation Loss)
        total_energy_pred = torch.sum(y_pred, dim=(1, 2))  # (batch_size,)
        total_energy_true = torch.sum(y_true, dim=(1, 2))  # (batch_size,)

        # 计算加权均方误差 (Weighted MSE Loss)
        squared_diff = (y_pred - y_true) ** 2  # (batch_size, H, W)
        weight = y_true  # 真实值作为权重 (batch_size, H, W)
        weighted_mse = torch.sum(weight * squared_diff, dim=(1, 2)) / (torch.sum(y_true, dim=(1, 2)) + 1e-8)  # 避免除零
        weighted_mse_loss = torch.mean(weighted_mse)  # 取 batch 维度的均值

        # 组合两个损失
        total_loss = weighted_mse_loss

        return total_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    # loss_function_compare = nn.MSELoss()
    # data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, (batch_X, batch_Y) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs=model(batch_X.to(device))
        loss_train=loss_function(outputs,batch_Y.to(device))
        # loss_compare = loss_function_compare(outputs, batch_Y.to(device))
        loss_train.backward()
        optimizer.step()
        
        mean_loss = (mean_loss * step + loss_train.detach()) / (step + 1)  # update mean losses
        # 打印平均loss
        # data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 7))
        
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
