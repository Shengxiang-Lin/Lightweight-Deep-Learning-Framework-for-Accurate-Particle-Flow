import torch
import torch.nn.functional as F
from torch import nn

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
CNN_5m:
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
           Conv2d-12          [-1, 512, 14, 14]       3,277,312
      BatchNorm2d-13          [-1, 512, 14, 14]           1,024
             ReLU-14          [-1, 512, 14, 14]               0
        MaxPool2d-15            [-1, 512, 7, 7]               0
  ConvTranspose2d-16          [-1, 256, 14, 14]         524,544
      BatchNorm2d-17          [-1, 256, 14, 14]             512
             ReLU-18          [-1, 256, 14, 14]               0
  ConvTranspose2d-19          [-1, 128, 28, 28]         131,200
      BatchNorm2d-20          [-1, 128, 28, 28]             256
             ReLU-21          [-1, 128, 28, 28]               0
  ConvTranspose2d-22           [-1, 64, 56, 56]          32,832
      BatchNorm2d-23           [-1, 64, 56, 56]             128
             ReLU-24           [-1, 64, 56, 56]               0
           Conv2d-25            [-1, 1, 56, 56]           1,601
          Sigmoid-26            [-1, 1, 56, 56]               0
================================================================
Total params: 5,001,153
Trainable params: 5,001,153
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 30.10
Params size (MB): 19.08
Estimated Total Size (MB): 49.22
----------------------------------------------------------------
'''''''''


class CNN_5m(nn.Module):
    def __init__(self):
        super(CNN_5m,self).__init__()
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
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
CNNwithSEBlock_5m:
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
           Conv2d-12          [-1, 512, 14, 14]       3,277,312
      BatchNorm2d-13          [-1, 512, 14, 14]           1,024
             ReLU-14          [-1, 512, 14, 14]               0
        MaxPool2d-15            [-1, 512, 7, 7]               0
           Linear-16                   [-1, 32]          16,384
           Linear-17                  [-1, 512]          16,384
          SEBlock-18            [-1, 512, 7, 7]               0
  ConvTranspose2d-19          [-1, 256, 14, 14]         524,544
      BatchNorm2d-20          [-1, 256, 14, 14]             512
             ReLU-21          [-1, 256, 14, 14]               0
  ConvTranspose2d-22          [-1, 128, 28, 28]         131,200
      BatchNorm2d-23          [-1, 128, 28, 28]             256
             ReLU-24          [-1, 128, 28, 28]               0
  ConvTranspose2d-25           [-1, 64, 56, 56]          32,832
      BatchNorm2d-26           [-1, 64, 56, 56]             128
             ReLU-27           [-1, 64, 56, 56]               0
           Conv2d-28            [-1, 1, 56, 56]           1,601
          Sigmoid-29            [-1, 1, 56, 56]               0
================================================================
Total params: 5,033,921
Trainable params: 5,033,921
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 30.29
Params size (MB): 19.20
Estimated Total Size (MB): 49.54
----------------------------------------------------------------
'''''''''''''''''''''''



class CNNwithSEBlock_5m(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock_5m,self).__init__()

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
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.se1=SEBlock(512)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
CNNwithSelfattention_5m:
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
           Conv2d-12          [-1, 512, 14, 14]       3,277,312
      BatchNorm2d-13          [-1, 512, 14, 14]           1,024
             ReLU-14          [-1, 512, 14, 14]               0
        MaxPool2d-15            [-1, 512, 7, 7]               0
           Conv2d-16            [-1, 512, 7, 7]         262,656
           Conv2d-17            [-1, 512, 7, 7]         262,656
           Conv2d-18            [-1, 512, 7, 7]         262,656
          Softmax-19               [-1, 49, 49]               0
    SelfAttention-20            [-1, 512, 7, 7]               0
  ConvTranspose2d-21          [-1, 256, 14, 14]         524,544
      BatchNorm2d-22          [-1, 256, 14, 14]             512
             ReLU-23          [-1, 256, 14, 14]               0
  ConvTranspose2d-24          [-1, 128, 28, 28]         131,200
      BatchNorm2d-25          [-1, 128, 28, 28]             256
             ReLU-26          [-1, 128, 28, 28]               0
  ConvTranspose2d-27           [-1, 64, 56, 56]          32,832
      BatchNorm2d-28           [-1, 64, 56, 56]             128
             ReLU-29           [-1, 64, 56, 56]               0
           Conv2d-30            [-1, 1, 56, 56]           1,601
          Sigmoid-31            [-1, 1, 56, 56]               0
================================================================
Total params: 5,789,121
Trainable params: 5,789,121
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 30.88
Params size (MB): 22.08
Estimated Total Size (MB): 53.01
----------------------------------------------------------------
'''''''''''

class CNNwithSelfattention_5m(nn.Module):
    def __init__(self):
        super(CNNwithSelfattention_5m,self).__init__()

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
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.attention = SelfAttention(512)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
CNN3D_5m:
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
           Conv2d-18          [-1, 512, 14, 14]       3,277,312
      BatchNorm2d-19          [-1, 512, 14, 14]           1,024
             ReLU-20          [-1, 512, 14, 14]               0
        MaxPool2d-21            [-1, 512, 7, 7]               0
  ConvTranspose2d-22          [-1, 256, 14, 14]         524,544
      BatchNorm2d-23          [-1, 256, 14, 14]             512
             ReLU-24          [-1, 256, 14, 14]               0
  ConvTranspose2d-25          [-1, 128, 28, 28]         131,200
      BatchNorm2d-26          [-1, 128, 28, 28]             256
             ReLU-27          [-1, 128, 28, 28]               0
  ConvTranspose2d-28           [-1, 64, 56, 56]          32,832
      BatchNorm2d-29           [-1, 64, 56, 56]             128
             ReLU-30           [-1, 64, 56, 56]               0
           Conv2d-31            [-1, 1, 56, 56]           1,601
          Sigmoid-32            [-1, 1, 56, 56]               0
================================================================
Total params: 5,014,961
Trainable params: 5,014,961
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 30.39
Params size (MB): 19.13
Estimated Total Size (MB): 49.56
----------------------------------------------------------------
'''

class CNN3D_5m(nn.Module):
    def __init__(self):
        super(CNN3D_5m,self).__init__()
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
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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
UNet_5m:
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
           Conv2d-22            [-1, 512, 7, 7]         524,800
        LeakyReLU-23            [-1, 512, 7, 7]               0
DownSampling_UNet-24            [-1, 512, 7, 7]               0
           Conv2d-25            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-26            [-1, 512, 7, 7]           1,024
          Dropout-27            [-1, 512, 7, 7]               0
        LeakyReLU-28            [-1, 512, 7, 7]               0
        Conv_UNet-29            [-1, 512, 7, 7]               0
           Conv2d-30          [-1, 256, 14, 14]         131,328
  UpSampling_UNet-31          [-1, 512, 14, 14]               0
           Conv2d-32          [-1, 256, 14, 14]       1,179,904
      BatchNorm2d-33          [-1, 256, 14, 14]             512
          Dropout-34          [-1, 256, 14, 14]               0
        LeakyReLU-35          [-1, 256, 14, 14]               0
        Conv_UNet-36          [-1, 256, 14, 14]               0
           Conv2d-37          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-38          [-1, 256, 28, 28]               0
           Conv2d-39          [-1, 128, 28, 28]         295,040
      BatchNorm2d-40          [-1, 128, 28, 28]             256
          Dropout-41          [-1, 128, 28, 28]               0
        LeakyReLU-42          [-1, 128, 28, 28]               0
        Conv_UNet-43          [-1, 128, 28, 28]               0
           Conv2d-44           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-45          [-1, 128, 56, 56]               0
           Conv2d-46           [-1, 64, 56, 56]          73,792
      BatchNorm2d-47           [-1, 64, 56, 56]             128
          Dropout-48           [-1, 64, 56, 56]               0
        LeakyReLU-49           [-1, 64, 56, 56]               0
        Conv_UNet-50           [-1, 64, 56, 56]               0
           Conv2d-51            [-1, 1, 56, 56]             577
          Sigmoid-52            [-1, 1, 56, 56]               0
================================================================
Total params: 5,513,473
Trainable params: 5,513,473
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 39.86
Params size (MB): 21.03
Estimated Total Size (MB): 60.94
----------------------------------------------------------------
'''''''''''

class UNet_5m(nn.Module):
    def __init__(self,in_channels):
        super(UNet_5m, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.D3 = DownSampling_UNet(256, 512)
        self.C4 = Conv_UNet(512, 512)
        self.U1 = UpSampling_UNet(512, 256)
        self.C5 = Conv_UNet(512, 256)        
        self.U2 = UpSampling_UNet(256, 128)
        self.C6 = Conv_UNet(256, 128)
        self.U3 = UpSampling_UNet(128, 64)
        self.C7 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))

'''''''''''
UnetwithSEBlock_5m:
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
           Conv2d-22            [-1, 512, 7, 7]         524,800
        LeakyReLU-23            [-1, 512, 7, 7]               0
DownSampling_UNet-24            [-1, 512, 7, 7]               0
           Conv2d-25            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-26            [-1, 512, 7, 7]           1,024
          Dropout-27            [-1, 512, 7, 7]               0
        LeakyReLU-28            [-1, 512, 7, 7]               0
        Conv_UNet-29            [-1, 512, 7, 7]               0
           Linear-30                   [-1, 32]          16,384
           Linear-31                  [-1, 512]          16,384
          SEBlock-32            [-1, 512, 7, 7]               0
           Conv2d-33          [-1, 256, 14, 14]         131,328
  UpSampling_UNet-34          [-1, 512, 14, 14]               0
           Conv2d-35          [-1, 256, 14, 14]       1,179,904
      BatchNorm2d-36          [-1, 256, 14, 14]             512
          Dropout-37          [-1, 256, 14, 14]               0
        LeakyReLU-38          [-1, 256, 14, 14]               0
        Conv_UNet-39          [-1, 256, 14, 14]               0
           Conv2d-40          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-41          [-1, 256, 28, 28]               0
           Conv2d-42          [-1, 128, 28, 28]         295,040
      BatchNorm2d-43          [-1, 128, 28, 28]             256
          Dropout-44          [-1, 128, 28, 28]               0
        LeakyReLU-45          [-1, 128, 28, 28]               0
        Conv_UNet-46          [-1, 128, 28, 28]               0
           Conv2d-47           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-48          [-1, 128, 56, 56]               0
           Conv2d-49           [-1, 64, 56, 56]          73,792
      BatchNorm2d-50           [-1, 64, 56, 56]             128
          Dropout-51           [-1, 64, 56, 56]               0
        LeakyReLU-52           [-1, 64, 56, 56]               0
        Conv_UNet-53           [-1, 64, 56, 56]               0
           Conv2d-54            [-1, 1, 56, 56]             577
          Sigmoid-55            [-1, 1, 56, 56]               0
================================================================
Total params: 5,546,241
Trainable params: 5,546,241
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 40.06
Params size (MB): 21.16
Estimated Total Size (MB): 61.26
----------------------------------------------------------------
'''''''''''
class UnetwithSEBlock_5m(nn.Module):
    def __init__(self,in_channels):
        super(UnetwithSEBlock_5m, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.D3 = DownSampling_UNet(256, 512)
        self.C4 = Conv_UNet(512, 512)
        self.se1=SEBlock(512)
        self.U1 = UpSampling_UNet(512, 256)
        self.C5 = Conv_UNet(512, 256)        
        self.U2 = UpSampling_UNet(256, 128)
        self.C6 = Conv_UNet(256, 128)
        self.U3 = UpSampling_UNet(128, 64)
        self.C7 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        R4=self.se1(R4)
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))

''''''''''''''''
UnetwithSelfattention_5m:
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
           Conv2d-22            [-1, 512, 7, 7]         524,800
        LeakyReLU-23            [-1, 512, 7, 7]               0
DownSampling_UNet-24            [-1, 512, 7, 7]               0
           Conv2d-25            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-26            [-1, 512, 7, 7]           1,024
          Dropout-27            [-1, 512, 7, 7]               0
        LeakyReLU-28            [-1, 512, 7, 7]               0
        Conv_UNet-29            [-1, 512, 7, 7]               0
           Conv2d-30            [-1, 512, 7, 7]         262,656
           Conv2d-31            [-1, 512, 7, 7]         262,656
           Conv2d-32            [-1, 512, 7, 7]         262,656
          Softmax-33               [-1, 49, 49]               0
    SelfAttention-34            [-1, 512, 7, 7]               0
           Conv2d-35          [-1, 256, 14, 14]         131,328
  UpSampling_UNet-36          [-1, 512, 14, 14]               0
           Conv2d-37          [-1, 256, 14, 14]       1,179,904
      BatchNorm2d-38          [-1, 256, 14, 14]             512
          Dropout-39          [-1, 256, 14, 14]               0
        LeakyReLU-40          [-1, 256, 14, 14]               0
        Conv_UNet-41          [-1, 256, 14, 14]               0
           Conv2d-42          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-43          [-1, 256, 28, 28]               0
           Conv2d-44          [-1, 128, 28, 28]         295,040
      BatchNorm2d-45          [-1, 128, 28, 28]             256
          Dropout-46          [-1, 128, 28, 28]               0
        LeakyReLU-47          [-1, 128, 28, 28]               0
        Conv_UNet-48          [-1, 128, 28, 28]               0
           Conv2d-49           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-50          [-1, 128, 56, 56]               0
           Conv2d-51           [-1, 64, 56, 56]          73,792
      BatchNorm2d-52           [-1, 64, 56, 56]             128
          Dropout-53           [-1, 64, 56, 56]               0
        LeakyReLU-54           [-1, 64, 56, 56]               0
        Conv_UNet-55           [-1, 64, 56, 56]               0
           Conv2d-56            [-1, 1, 56, 56]             577
          Sigmoid-57            [-1, 1, 56, 56]               0
================================================================
Total params: 6,301,441
Trainable params: 6,301,441
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 40.64
Params size (MB): 24.04
Estimated Total Size (MB): 64.73
----------------------------------------------------------------
'''''''''''''''
class UnetwithSelfattention_5m(nn.Module):
    def __init__(self,in_channels):
        super(UnetwithSelfattention_5m, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.D3 = DownSampling_UNet(256, 512)
        self.C4 = Conv_UNet(512, 512)
        self.attention = SelfAttention(512)
        self.U1 = UpSampling_UNet(512, 256)
        self.C5 = Conv_UNet(512, 256)        
        self.U2 = UpSampling_UNet(256, 128)
        self.C6 = Conv_UNet(256, 128)
        self.U3 = UpSampling_UNet(128, 64)
        self.C7 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        R4=self.attention(R4)
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))



'''''''''''''''''
Unet3D_5m:
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
           Conv2d-28            [-1, 512, 7, 7]         524,800
        LeakyReLU-29            [-1, 512, 7, 7]               0
DownSampling_UNet-30            [-1, 512, 7, 7]               0
           Conv2d-31            [-1, 512, 7, 7]       2,359,808
      BatchNorm2d-32            [-1, 512, 7, 7]           1,024
          Dropout-33            [-1, 512, 7, 7]               0
        LeakyReLU-34            [-1, 512, 7, 7]               0
        Conv_UNet-35            [-1, 512, 7, 7]               0
           Conv2d-36          [-1, 256, 14, 14]         131,328
  UpSampling_UNet-37          [-1, 512, 14, 14]               0
           Conv2d-38          [-1, 256, 14, 14]       1,179,904
      BatchNorm2d-39          [-1, 256, 14, 14]             512
          Dropout-40          [-1, 256, 14, 14]               0
        LeakyReLU-41          [-1, 256, 14, 14]               0
        Conv_UNet-42          [-1, 256, 14, 14]               0
           Conv2d-43          [-1, 128, 28, 28]          32,896
  UpSampling_UNet-44          [-1, 256, 28, 28]               0
           Conv2d-45          [-1, 128, 28, 28]         295,040
      BatchNorm2d-46          [-1, 128, 28, 28]             256
          Dropout-47          [-1, 128, 28, 28]               0
        LeakyReLU-48          [-1, 128, 28, 28]               0
        Conv_UNet-49          [-1, 128, 28, 28]               0
           Conv2d-50           [-1, 64, 56, 56]           8,256
  UpSampling_UNet-51          [-1, 128, 56, 56]               0
           Conv2d-52           [-1, 64, 56, 56]          73,792
      BatchNorm2d-53           [-1, 64, 56, 56]             128
          Dropout-54           [-1, 64, 56, 56]               0
        LeakyReLU-55           [-1, 64, 56, 56]               0
        Conv_UNet-56           [-1, 64, 56, 56]               0
           Conv2d-57            [-1, 1, 56, 56]             577
          Sigmoid-58            [-1, 1, 56, 56]               0
          UNet_5m-59            [-1, 1, 56, 56]               0
================================================================
Total params: 5,519,089
Trainable params: 5,519,089
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 40.17
Params size (MB): 21.05
Estimated Total Size (MB): 61.27
----------------------------------------------------------------
'''''''''''''''''

class Unet3D_5m(nn.Module):
    def __init__(self):
        super(Unet3D_5m,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1)) #(batch_size,2,1,56,56)
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2)) #(batch_size,2,1,56,56)
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3)) #(batch_size,2,1,56,56)
        self.unet=UNet_5m(12)
    
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




class UNetall_5m(nn.Module):
    def __init__(self):
        super(UNetall_5m, self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1)) #(batch_size,2,1,56,56)
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2)) #(batch_size,2,1,56,56)
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3)) #(batch_size,2,1,56,56)
        self.C1 = Conv_UNet(12, 64)
        self.sel1=SEBlock(64)
        self.D1 = DownSampling_UNet(64, 128)
        self.C2 = Conv_UNet(128, 128)
        self.sel2=SEBlock(128)
        self.D2 = DownSampling_UNet(128, 256)
        self.C3 = Conv_UNet(256, 256)
        self.sel3=SEBlock(256)
        self.D3 = DownSampling_UNet(256, 512)
        self.C4 = Conv_UNet(512, 512)
        self.sel4=SEBlock(512)
        self.attention = SelfAttention(512)
        self.U1 = UpSampling_UNet(512, 256)
        self.C5 = Conv_UNet(512, 256)
        self.U2 = UpSampling_UNet(256, 128)
        self.C6 = Conv_UNet(256, 128)
        self.U3 = UpSampling_UNet(128, 64)
        self.C7 = Conv_UNet(128, 64)
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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

        R1 = self.sel1(self.C1(x))
        R2 = self.sel2(self.C2(self.D1(R1)))
        R3 = self.sel3(self.C3(self.D2(R2)))
        R4 = self.sel4(self.C4(self.D3(R3)))
        R4=self.attention(R4)
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))