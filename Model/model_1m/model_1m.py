import torch
import torch.nn.functional as F
from torch import nn

'''''''''''''''''''''
该文件中的网络均使用以下学习率调度器:
def lf_function(epoch): 
    if epoch < warmup_epochs_1:
        return 1
    else: 
        return 0.1
'''''''''''''''''''''


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1) #H*W->H*W
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # 生成查询、键、值
        queries = self.query(x).view(batch_size, C, -1) # (B, C, H*W)
        keys = self.key(x).view(batch_size, C, -1) # (B, C, H*W)
        values = self.value(x).view(batch_size, C, -1) # (B, C, H*W)

        # 计算自注意力
        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys) # (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)

        out = torch.bmm(values, attention_scores.permute(0, 2, 1)) # (B, C, H*W)
        return out.view(batch_size, C, H, W) #不改变形状




class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock,self).__init__()
        self.fc1=nn.Linear(channels,channels//reduction,bias=False)
        self.fc2=nn.Linear(channels//reduction,channels,bias=False)

    def forward(self,x):
        b, c,_,_=x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c) # Squeeze
        y=F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1) # Excitation - 2nd layer
        return x * y.expand_as(x) # Scale

class DownSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=2, stride=2),  # 2x2卷积，步幅2会让特征尺寸减半
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)
#定义上采样层
class UpSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C_in, C_out, kernel_size=1)  # 改变通道数的卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 使用最近邻插值进行上采样
        x = self.Up(up)  # 改变输出通道数
        x = torch.cat([x, r], dim=1)  # 进行跳跃连接，拼接特征
        return x


class Conv_UNet(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv_UNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1),  # 3x3卷积，padding=1保持尺寸不变
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class DownSampling_UNet(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling_UNet, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=2, stride=2),  # 2x2卷积，步幅2会让特征尺寸减半
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)

class UpSampling_UNet(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling_UNet, self).__init__()
        self.Up = nn.Conv2d(C_in, C_out, kernel_size=1)  # 改变通道数的卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 使用最近邻插值进行上采样
        x = self.Up(up)  # 改变输出通道数
        x = torch.cat([x, r], dim=1)  # 进行跳跃连接，拼接特征
        return x
        

'''''''''
CNN_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]           6,464
       BatchNorm2d-2           [-1, 64, 56, 56]             128
              ReLU-3           [-1, 64, 56, 56]               0
            Conv2d-4          [-1, 128, 56, 56]         204,928
       BatchNorm2d-5          [-1, 128, 56, 56]             256
              ReLU-6          [-1, 128, 56, 56]               0
         MaxPool2d-7          [-1, 128, 28, 28]               0
            Conv2d-8          [-1, 256, 28, 28]         819,456
       BatchNorm2d-9          [-1, 256, 28, 28]             512
             ReLU-10          [-1, 256, 28, 28]               0
        MaxPool2d-11          [-1, 256, 14, 14]               0
  ConvTranspose2d-12          [-1, 128, 28, 28]         131,200
      BatchNorm2d-13          [-1, 128, 28, 28]             256
             ReLU-14          [-1, 128, 28, 28]               0
  ConvTranspose2d-15           [-1, 64, 56, 56]          32,832
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
           Conv2d-18            [-1, 1, 56, 56]           1,601
          Sigmoid-19            [-1, 1, 56, 56]               0
================================================================
Total params: 1,197,761
Trainable params: 1,197,761
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 26.46
Params size (MB): 4.57
Estimated Total Size (MB): 31.08
----------------------------------------------------------------
'''''''''

class CNN_1m(nn.Module):
    def __init__(self):
        super(CNN_1m,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x


'''''''''''''''''''''
CNNwithSEBlock_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]           6,464
       BatchNorm2d-2           [-1, 64, 56, 56]             128
              ReLU-3           [-1, 64, 56, 56]               0
            Conv2d-4          [-1, 128, 56, 56]         204,928
       BatchNorm2d-5          [-1, 128, 56, 56]             256
              ReLU-6          [-1, 128, 56, 56]               0
         MaxPool2d-7          [-1, 128, 28, 28]               0
            Conv2d-8          [-1, 256, 28, 28]         819,456
       BatchNorm2d-9          [-1, 256, 28, 28]             512
             ReLU-10          [-1, 256, 28, 28]               0
        MaxPool2d-11          [-1, 256, 14, 14]               0
           Linear-12                   [-1, 16]           4,096
           Linear-13                  [-1, 256]           4,096
          SEBlock-14          [-1, 256, 14, 14]               0
  ConvTranspose2d-15          [-1, 128, 28, 28]         131,200
      BatchNorm2d-16          [-1, 128, 28, 28]             256
             ReLU-17          [-1, 128, 28, 28]               0
  ConvTranspose2d-18           [-1, 64, 56, 56]          32,832
      BatchNorm2d-19           [-1, 64, 56, 56]             128
             ReLU-20           [-1, 64, 56, 56]               0
           Conv2d-21            [-1, 1, 56, 56]           1,601
          Sigmoid-22            [-1, 1, 56, 56]               0
================================================================
Total params: 1,205,953
Trainable params: 1,205,953
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 26.85
Params size (MB): 4.60
Estimated Total Size (MB): 31.49
----------------------------------------------------------------
'''''''''''''''''''''''

class CNNwithSEBlock_1m(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock_1m,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        )
        self.se1=SEBlock(256)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.se1(x)
        x=self.decoder(x)
        return x




'''''''''''
CNNwithSelfattention_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]           6,464
       BatchNorm2d-2           [-1, 64, 56, 56]             128
              ReLU-3           [-1, 64, 56, 56]               0
            Conv2d-4          [-1, 128, 56, 56]         204,928
       BatchNorm2d-5          [-1, 128, 56, 56]             256
              ReLU-6          [-1, 128, 56, 56]               0
         MaxPool2d-7          [-1, 128, 28, 28]               0
            Conv2d-8          [-1, 256, 28, 28]         819,456
       BatchNorm2d-9          [-1, 256, 28, 28]             512
             ReLU-10          [-1, 256, 28, 28]               0
        MaxPool2d-11          [-1, 256, 14, 14]               0
           Conv2d-12          [-1, 256, 14, 14]          65,792
           Conv2d-13          [-1, 256, 14, 14]          65,792
           Conv2d-14          [-1, 256, 14, 14]          65,792
          Softmax-15             [-1, 196, 196]               0
    SelfAttention-16          [-1, 256, 14, 14]               0
  ConvTranspose2d-17          [-1, 128, 28, 28]         131,200
      BatchNorm2d-18          [-1, 128, 28, 28]             256
             ReLU-19          [-1, 128, 28, 28]               0
  ConvTranspose2d-20           [-1, 64, 56, 56]          32,832
      BatchNorm2d-21           [-1, 64, 56, 56]             128
             ReLU-22           [-1, 64, 56, 56]               0
           Conv2d-23            [-1, 1, 56, 56]           1,601
          Sigmoid-24            [-1, 1, 56, 56]               0
================================================================
Total params: 1,395,137
Trainable params: 1,395,137
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 28.29
Params size (MB): 5.32
Estimated Total Size (MB): 33.66
----------------------------------------------------------------
'''''''''''

class CNNwithSelfattention_1m(nn.Module):
    def __init__(self):
        super(CNNwithSelfattention_1m,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        )
        self.attention = SelfAttention(256)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.attention(x)
        x=self.decoder(x)
        return x

''''''''''
CNN3D_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 2, 1, 56, 56]              56
            Conv3d-2         [-1, 2, 1, 56, 56]             152
            Conv3d-3         [-1, 2, 1, 56, 56]             296
            Conv3d-4         [-1, 2, 1, 56, 56]              56
            Conv3d-5         [-1, 2, 1, 56, 56]             152
            Conv3d-6         [-1, 2, 1, 56, 56]             296
            Conv2d-7           [-1, 64, 56, 56]          19,264
       BatchNorm2d-8           [-1, 64, 56, 56]             128
              ReLU-9           [-1, 64, 56, 56]               0
           Conv2d-10          [-1, 128, 56, 56]         204,928
      BatchNorm2d-11          [-1, 128, 56, 56]             256
             ReLU-12          [-1, 128, 56, 56]               0
        MaxPool2d-13          [-1, 128, 28, 28]               0
           Conv2d-14          [-1, 256, 28, 28]         819,456
      BatchNorm2d-15          [-1, 256, 28, 28]             512
             ReLU-16          [-1, 256, 28, 28]               0
        MaxPool2d-17          [-1, 256, 14, 14]               0
  ConvTranspose2d-18          [-1, 128, 28, 28]         131,200
      BatchNorm2d-19          [-1, 128, 28, 28]             256
             ReLU-20          [-1, 128, 28, 28]               0
  ConvTranspose2d-21           [-1, 64, 56, 56]          32,832
      BatchNorm2d-22           [-1, 64, 56, 56]             128
             ReLU-23           [-1, 64, 56, 56]               0
           Conv2d-24            [-1, 1, 56, 56]           1,601
          Sigmoid-25            [-1, 1, 56, 56]               0
================================================================
Total params: 1,211,569
Trainable params: 1,211,569
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 26.75
Params size (MB): 4.62
Estimated Total Size (MB): 31.42
----------------------------------------------------------------
'''

class CNN3D_1m(nn.Module):
    def __init__(self):
        super(CNN3D_1m,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   # 14x14 -> 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 28x28 -> 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x = x.unsqueeze(1)
        x_e_h_n = x[:,:,:3,:,:]
        x_e_h_p = x[:,:,[0,1,3],:,:]
        x2 = self.conv3x3x3(x_e_h_n)
        x3 = self.conv3x5x5(x_e_h_n)
        x4 = self.conv3x7x7(x_e_h_n)
        x5 = self.conv3x3x3(x_e_h_p)
        x6 = self.conv3x5x5(x_e_h_p)
        x7 = self.conv3x7x7(x_e_h_p)
        x = torch.cat((x2,x3,x4,x5,x6,x7),dim=1).view(-1,12,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x

'''''''''''
UNet_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]           2,368
       BatchNorm2d-2           [-1, 64, 56, 56]             128
           Dropout-3           [-1, 64, 56, 56]               0
         LeakyReLU-4           [-1, 64, 56, 56]               0
         Conv_UNet-5           [-1, 64, 56, 56]               0
            Conv2d-6          [-1, 128, 28, 28]          32,896
         LeakyReLU-7          [-1, 128, 28, 28]               0
 DownSampling_UNet-8          [-1, 128, 28, 28]               0
            Conv2d-9          [-1, 128, 28, 28]         147,584
      BatchNorm2d-10          [-1, 128, 28, 28]             256
          Dropout-11          [-1, 128, 28, 28]               0
        LeakyReLU-12          [-1, 128, 28, 28]               0
        Conv_UNet-13          [-1, 128, 28, 28]               0
           Conv2d-14          [-1, 256, 14, 14]         131,328
        LeakyReLU-15          [-1, 256, 14, 14]               0
DownSampling_UNet-16          [-1, 256, 14, 14]               0
           Conv2d-17          [-1, 256, 14, 14]         590,080
      BatchNorm2d-18          [-1, 256, 14, 14]             512
          Dropout-19          [-1, 256, 14, 14]               0
        LeakyReLU-20          [-1, 256, 14, 14]               0
        Conv_UNet-21          [-1, 256, 14, 14]               0
           Conv2d-22          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-23          [-1, 256, 28, 28]               0
           Conv2d-24          [-1, 128, 28, 28]         295,040
      BatchNorm2d-25          [-1, 128, 28, 28]             256
          Dropout-26          [-1, 128, 28, 28]               0
        LeakyReLU-27          [-1, 128, 28, 28]               0
        Conv_UNet-28          [-1, 128, 28, 28]               0
           Conv2d-29           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-30          [-1, 128, 56, 56]               0
           Conv2d-31           [-1, 64, 56, 56]          73,792
      BatchNorm2d-32           [-1, 64, 56, 56]             128
          Dropout-33           [-1, 64, 56, 56]               0
        LeakyReLU-34           [-1, 64, 56, 56]               0
        Conv_UNet-35           [-1, 64, 56, 56]               0
           Conv2d-36            [-1, 1, 56, 56]             577
          Sigmoid-37            [-1, 1, 56, 56]               0
================================================================
Total params: 1,316,097
Trainable params: 1,316,097
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 35.27
Params size (MB): 5.02
Estimated Total Size (MB): 40.33
----------------------------------------------------------------
'''''''''''

class UNet_1m(nn.Module):
    def __init__(self,in_channels):
        super(UNet_1m, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.U1 = UpSampling_UNet(256, 128)
        self.C4 = Conv_UNet(256, 128)
        self.U2 = UpSampling_UNet(128, 64)
        self.C5 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        up1 = self.C4(self.U1(R3, R2))
        c = self.C5(self.U2(up1,R1))
        return self.sigmoid(self.pred(c))

'''''''''''
UnetwithSEBlock_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]           2,368
       BatchNorm2d-2           [-1, 64, 56, 56]             128
           Dropout-3           [-1, 64, 56, 56]               0
         LeakyReLU-4           [-1, 64, 56, 56]               0
         Conv_UNet-5           [-1, 64, 56, 56]               0
            Conv2d-6          [-1, 128, 28, 28]          32,896
         LeakyReLU-7          [-1, 128, 28, 28]               0
 DownSampling_UNet-8          [-1, 128, 28, 28]               0
            Conv2d-9          [-1, 128, 28, 28]         147,584
      BatchNorm2d-10          [-1, 128, 28, 28]             256
          Dropout-11          [-1, 128, 28, 28]               0
        LeakyReLU-12          [-1, 128, 28, 28]               0
        Conv_UNet-13          [-1, 128, 28, 28]               0
           Conv2d-14          [-1, 256, 14, 14]         131,328
        LeakyReLU-15          [-1, 256, 14, 14]               0
DownSampling_UNet-16          [-1, 256, 14, 14]               0
           Conv2d-17          [-1, 256, 14, 14]         590,080
      BatchNorm2d-18          [-1, 256, 14, 14]             512
          Dropout-19          [-1, 256, 14, 14]               0
        LeakyReLU-20          [-1, 256, 14, 14]               0
        Conv_UNet-21          [-1, 256, 14, 14]               0
           Linear-22                   [-1, 16]           4,096
           Linear-23                  [-1, 256]           4,096
          SEBlock-24          [-1, 256, 14, 14]               0
           Conv2d-25          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-26          [-1, 256, 28, 28]               0
           Conv2d-27          [-1, 128, 28, 28]         295,040
      BatchNorm2d-28          [-1, 128, 28, 28]             256
          Dropout-29          [-1, 128, 28, 28]               0
        LeakyReLU-30          [-1, 128, 28, 28]               0
        Conv_UNet-31          [-1, 128, 28, 28]               0
           Conv2d-32           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-33          [-1, 128, 56, 56]               0
           Conv2d-34           [-1, 64, 56, 56]          73,792
      BatchNorm2d-35           [-1, 64, 56, 56]             128
          Dropout-36           [-1, 64, 56, 56]               0
        LeakyReLU-37           [-1, 64, 56, 56]               0
        Conv_UNet-38           [-1, 64, 56, 56]               0
           Conv2d-39            [-1, 1, 56, 56]             577
          Sigmoid-40            [-1, 1, 56, 56]               0
================================================================
Total params: 1,324,289
Trainable params: 1,324,289
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 35.65
Params size (MB): 5.05
Estimated Total Size (MB): 40.75
----------------------------------------------------------------
'''''''''''
class UnetwithSEBlock_1m(nn.Module):
    def __init__(self,in_channels):
        super(UnetwithSEBlock_1m, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.se1=SEBlock(256)
        self.U1 = UpSampling_UNet(256, 128)
        self.C4 = Conv_UNet(256, 128)
        self.U2 = UpSampling_UNet(128, 64)
        self.C5 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R3=self.se1(R3)
        up1 = self.C4(self.U1(R3, R2))
        c = self.C5(self.U2(up1,R1))
        return self.sigmoid(self.pred(c))

''''''''''''''''
UnetwithSelfattention_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]           2,368
       BatchNorm2d-2           [-1, 64, 56, 56]             128
           Dropout-3           [-1, 64, 56, 56]               0
         LeakyReLU-4           [-1, 64, 56, 56]               0
         Conv_UNet-5           [-1, 64, 56, 56]               0
            Conv2d-6          [-1, 128, 28, 28]          32,896
         LeakyReLU-7          [-1, 128, 28, 28]               0
 DownSampling_UNet-8          [-1, 128, 28, 28]               0
            Conv2d-9          [-1, 128, 28, 28]         147,584
      BatchNorm2d-10          [-1, 128, 28, 28]             256
          Dropout-11          [-1, 128, 28, 28]               0
        LeakyReLU-12          [-1, 128, 28, 28]               0
        Conv_UNet-13          [-1, 128, 28, 28]               0
           Conv2d-14          [-1, 256, 14, 14]         131,328
        LeakyReLU-15          [-1, 256, 14, 14]               0
DownSampling_UNet-16          [-1, 256, 14, 14]               0
           Conv2d-17          [-1, 256, 14, 14]         590,080
      BatchNorm2d-18          [-1, 256, 14, 14]             512
          Dropout-19          [-1, 256, 14, 14]               0
        LeakyReLU-20          [-1, 256, 14, 14]               0
        Conv_UNet-21          [-1, 256, 14, 14]               0
           Conv2d-22          [-1, 256, 14, 14]          65,792
           Conv2d-23          [-1, 256, 14, 14]          65,792
           Conv2d-24          [-1, 256, 14, 14]          65,792
          Softmax-25             [-1, 196, 196]               0
    SelfAttention-26          [-1, 256, 14, 14]               0
           Conv2d-27          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-28          [-1, 256, 28, 28]               0
           Conv2d-29          [-1, 128, 28, 28]         295,040
      BatchNorm2d-30          [-1, 128, 28, 28]             256
          Dropout-31          [-1, 128, 28, 28]               0
        LeakyReLU-32          [-1, 128, 28, 28]               0
        Conv_UNet-33          [-1, 128, 28, 28]               0
           Conv2d-34           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-35          [-1, 128, 56, 56]               0
           Conv2d-36           [-1, 64, 56, 56]          73,792
      BatchNorm2d-37           [-1, 64, 56, 56]             128
          Dropout-38           [-1, 64, 56, 56]               0
        LeakyReLU-39           [-1, 64, 56, 56]               0
        Conv_UNet-40           [-1, 64, 56, 56]               0
           Conv2d-41            [-1, 1, 56, 56]             577
          Sigmoid-42            [-1, 1, 56, 56]               0
================================================================
Total params: 1,513,473
Trainable params: 1,513,473
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 37.09
Params size (MB): 5.77
Estimated Total Size (MB): 42.91
----------------------------------------------------------------
'''''''''''''''
class UnetwithSelfattention_1m(nn.Module):
    def __init__(self,in_channels):
        super(UnetwithSelfattention_1m, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.attention=SelfAttention(256)
        self.U1 = UpSampling_UNet(256, 128)
        self.C4 = Conv_UNet(256, 128)
        self.U2 = UpSampling_UNet(128, 64)
        self.C5 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R3=self.attention(R3)
        up1 = self.C4(self.U1(R3, R2))
        c = self.C5(self.U2(up1,R1))
        return self.sigmoid(self.pred(c))

'''''''''''''''''
Unet3D_1m:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 2, 1, 56, 56]              56
            Conv3d-2         [-1, 2, 1, 56, 56]             152
            Conv3d-3         [-1, 2, 1, 56, 56]             296
            Conv3d-4         [-1, 2, 1, 56, 56]              56
            Conv3d-5         [-1, 2, 1, 56, 56]             152
            Conv3d-6         [-1, 2, 1, 56, 56]             296
            Conv2d-7           [-1, 64, 56, 56]           6,976
       BatchNorm2d-8           [-1, 64, 56, 56]             128
           Dropout-9           [-1, 64, 56, 56]               0
        LeakyReLU-10           [-1, 64, 56, 56]               0
        Conv_UNet-11           [-1, 64, 56, 56]               0
           Conv2d-12          [-1, 128, 28, 28]          32,896
        LeakyReLU-13          [-1, 128, 28, 28]               0
DownSampling_UNet-14          [-1, 128, 28, 28]               0
           Conv2d-15          [-1, 128, 28, 28]         147,584
      BatchNorm2d-16          [-1, 128, 28, 28]             256
          Dropout-17          [-1, 128, 28, 28]               0
        LeakyReLU-18          [-1, 128, 28, 28]               0
        Conv_UNet-19          [-1, 128, 28, 28]               0
           Conv2d-20          [-1, 256, 14, 14]         131,328
        LeakyReLU-21          [-1, 256, 14, 14]               0
DownSampling_UNet-22          [-1, 256, 14, 14]               0
           Conv2d-23          [-1, 256, 14, 14]         590,080
      BatchNorm2d-24          [-1, 256, 14, 14]             512
          Dropout-25          [-1, 256, 14, 14]               0
        LeakyReLU-26          [-1, 256, 14, 14]               0
        Conv_UNet-27          [-1, 256, 14, 14]               0
           Conv2d-28          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-29          [-1, 256, 28, 28]               0
           Conv2d-30          [-1, 128, 28, 28]         295,040
      BatchNorm2d-31          [-1, 128, 28, 28]             256
          Dropout-32          [-1, 128, 28, 28]               0
        LeakyReLU-33          [-1, 128, 28, 28]               0
        Conv_UNet-34          [-1, 128, 28, 28]               0
           Conv2d-35           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-36          [-1, 128, 56, 56]               0
           Conv2d-37           [-1, 64, 56, 56]          73,792
      BatchNorm2d-38           [-1, 64, 56, 56]             128
          Dropout-39           [-1, 64, 56, 56]               0
        LeakyReLU-40           [-1, 64, 56, 56]               0
        Conv_UNet-41           [-1, 64, 56, 56]               0
           Conv2d-42            [-1, 1, 56, 56]             577
          Sigmoid-43            [-1, 1, 56, 56]               0
          UNet_1m-44            [-1, 1, 56, 56]               0
================================================================
Total params: 1,321,713
Trainable params: 1,321,713
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 35.58
Params size (MB): 5.04
Estimated Total Size (MB): 40.67
----------------------------------------------------------------
'''''''''''''''''

class Unet3D_1m(nn.Module):
    def __init__(self):
        super(Unet3D_1m,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1)) #(batch_size,2,1,56,56)
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2)) #(batch_size,2,1,56,56)
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3)) #(batch_size,2,1,56,56)
        self.unet=UNet_1m(12)
    
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x = x.unsqueeze(1)
        x_e_h_n = x[:,:,:3,:,:]
        x_e_h_p = x[:,:,[0,1,3],:,:]
        x2 = self.conv3x3x3(x_e_h_n)
        x3 = self.conv3x5x5(x_e_h_n)
        x4 = self.conv3x7x7(x_e_h_n)
        x5 = self.conv3x3x3(x_e_h_p)
        x6 = self.conv3x5x5(x_e_h_p)
        x7 = self.conv3x7x7(x_e_h_p)
        x = torch.cat((x2,x3,x4,x5,x6,x7),dim=1).view(-1,12,56,56) #(batch_size,12,56,56)
        x=self.unet(x)
        
        return x

