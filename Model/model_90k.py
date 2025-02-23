import torch
import torch.nn.functional as F
from torch import nn

'''''''''''''''''''''
该文件中的网络均使用以下学习率调度器:
def lf_function(epoch): 
    if epoch < warmup_epochs_1:
        return 1
    elif epoch < warmup_epochs_2: 
        return 0.1
    elif epoch < warmup_epochs_3:
        return((epoch - warmup_epochs_2) / (warmup_epochs_3 - warmup_epochs_2)) * 0.5 + 0.1
    else:
        return(((1 + math.cos((epoch - warmup_epochs_3) * math.pi / (args.epochs - warmup_epochs_3))) / 2) * 0.5 + 0.1)
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
CNN_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 56, 56]           1,616
       BatchNorm2d-2           [-1, 16, 56, 56]              32
              ReLU-3           [-1, 16, 56, 56]               0
            Conv2d-4           [-1, 32, 56, 56]          12,832
       BatchNorm2d-5           [-1, 32, 56, 56]              64
              ReLU-6           [-1, 32, 56, 56]               0
            Conv2d-7           [-1, 64, 56, 56]          51,264
       BatchNorm2d-8           [-1, 64, 56, 56]             128
              ReLU-9           [-1, 64, 56, 56]               0
           Conv2d-10           [-1, 16, 56, 56]          25,616
           Conv2d-11            [-1, 1, 56, 56]             401
          Sigmoid-12            [-1, 1, 56, 56]               0
================================================================
Total params: 91,953
Trainable params: 91,953
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 8.47
Params size (MB): 0.35
Estimated Total Size (MB): 8.87
----------------------------------------------------------------
None
'''''''''

class CNN_90k(nn.Module):
    def __init__(self):
        super(CNN_90k,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x


class CNN_withpool_90k(nn.Module):
    def __init__(self):
        super(CNN_withpool_90k,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x

'''''''''''''''''''''
CNNwithSEBlock_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 56, 56]           1,616
       BatchNorm2d-2           [-1, 16, 56, 56]              32
              ReLU-3           [-1, 16, 56, 56]               0
            Conv2d-4           [-1, 32, 56, 56]          12,832
       BatchNorm2d-5           [-1, 32, 56, 56]              64
              ReLU-6           [-1, 32, 56, 56]               0
            Conv2d-7           [-1, 64, 56, 56]          51,264
       BatchNorm2d-8           [-1, 64, 56, 56]             128
              ReLU-9           [-1, 64, 56, 56]               0
           Linear-10                    [-1, 4]             256
           Linear-11                   [-1, 64]             256
          SEBlock-12           [-1, 64, 56, 56]               0
           Conv2d-13           [-1, 16, 56, 56]          25,616
           Conv2d-14            [-1, 1, 56, 56]             401
          Sigmoid-15            [-1, 1, 56, 56]               0
================================================================
Total params: 92,465
Trainable params: 92,465
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 10.00
Params size (MB): 0.35
Estimated Total Size (MB): 10.40
----------------------------------------------------------------
'''''''''''''''''''''''



class CNNwithSEBlock_90k(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock_90k,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.se1=SEBlock(64)
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.se1(x)
        x=self.decoder(x)
        return x

class CNNwithSEBlock_withpool_90k(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock_withpool_90k,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.se1=SEBlock(64)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.se1(x)
        x=self.decoder(x)
        return x



'''''''''''
CNNwithSelfattention_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 56, 56]           1,616
       BatchNorm2d-2           [-1, 16, 56, 56]              32
              ReLU-3           [-1, 16, 56, 56]               0
            Conv2d-4           [-1, 32, 56, 56]          12,832
       BatchNorm2d-5           [-1, 32, 56, 56]              64
              ReLU-6           [-1, 32, 56, 56]               0
            Conv2d-7           [-1, 64, 56, 56]          51,264
       BatchNorm2d-8           [-1, 64, 56, 56]             128
              ReLU-9           [-1, 64, 56, 56]               0
           Conv2d-10           [-1, 64, 56, 56]           4,160
           Conv2d-11           [-1, 64, 56, 56]           4,160
           Conv2d-12           [-1, 64, 56, 56]           4,160
          Softmax-13           [-1, 3136, 3136]               0
    SelfAttention-14           [-1, 64, 56, 56]               0
           Conv2d-15           [-1, 16, 56, 56]          25,616
           Conv2d-16            [-1, 1, 56, 56]             401
          Sigmoid-17            [-1, 1, 56, 56]               0
================================================================
Total params: 104,433
Trainable params: 104,433
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 89.63
Params size (MB): 0.40
Estimated Total Size (MB): 90.07
----------------------------------------------------------------
'''''''''''

class CNNwithSelfattention_90k(nn.Module):
    def __init__(self):
        super(CNNwithSelfattention_90k,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.attention = SelfAttention(64)
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.attention(x)
        x=self.decoder(x)
        return x
    

class CNNwithSelfattention_withpool_90k(nn.Module):
    def __init__(self):
        super(CNNwithSelfattention_withpool_90k,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.attention = SelfAttention(64)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.attention(x)
        x=self.decoder(x)
        return x

''''''''''
CNN3D_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 2, 1, 56, 56]              56
            Conv3d-2         [-1, 2, 1, 56, 56]             152
            Conv3d-3         [-1, 2, 1, 56, 56]             296
            Conv3d-4         [-1, 2, 1, 56, 56]              56
            Conv3d-5         [-1, 2, 1, 56, 56]             152
            Conv3d-6         [-1, 2, 1, 56, 56]             296
            Conv2d-7           [-1, 32, 56, 56]           9,632
       BatchNorm2d-8           [-1, 32, 56, 56]              64
              ReLU-9           [-1, 32, 56, 56]               0
           Conv2d-10           [-1, 64, 56, 56]          51,264
      BatchNorm2d-11           [-1, 64, 56, 56]             128
             ReLU-12           [-1, 64, 56, 56]               0
           Conv2d-13           [-1, 16, 56, 56]          25,616
           Conv2d-14            [-1, 1, 56, 56]             401
          Sigmoid-15            [-1, 1, 56, 56]               0
================================================================
Total params: 88,113
Trainable params: 88,113
Non-trainable params: 0
----------------------------------------------------------------
'''

class CNN3D_90k(nn.Module):
    def __init__(self):
        super(CNN3D_90k,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
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


class CNN3Dver2_90k(nn.Module):
    def __init__(self):
        super(CNN3Dver2_90k,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.conv2x3x3 = nn.Conv3d(1, 2, kernel_size=(2,3,3), padding=(0,1,1))
        self.conv2x5x5 = nn.Conv3d(1, 2, kernel_size=(2,5,5), padding=(0,2,2))
        self.conv2x7x7 = nn.Conv3d(1, 2, kernel_size=(2,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x = x.unsqueeze(1)
        x_e_h_n = x[:,:,:3,:,:]
        x_e_h_p = x[:,:,[0,1,3],:,:]
        x_e_h = x[:,:,:2,:,:]
        x2 = self.conv3x3x3(x_e_h_n)
        x3 = self.conv3x5x5(x_e_h_n)
        x4 = self.conv3x7x7(x_e_h_n)
        x5 = self.conv3x3x3(x_e_h_p)
        x6 = self.conv3x5x5(x_e_h_p)
        x7 = self.conv3x7x7(x_e_h_p)
        x8 = self.conv2x3x3(x_e_h)
        x9 = self.conv2x5x5(x_e_h)
        x10 = self.conv2x7x7(x_e_h)
        x = torch.cat((x2,x3,x4,x5,x6,x7,x8,x9,x10),dim=1).view(-1,18,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x

class CNN3Dver3_90k(nn.Module):
    def __init__(self):
        super(CNN3Dver3_90k,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.conv2x3x3 = nn.Conv3d(1, 2, kernel_size=(2,3,3), padding=(0,1,1))
        self.conv2x5x5 = nn.Conv3d(1, 2, kernel_size=(2,5,5), padding=(0,2,2))
        self.conv2x7x7 = nn.Conv3d(1, 2, kernel_size=(2,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        self.fc = conv = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x = x.unsqueeze(1)
        x_e_h_n = x[:,:,:3,:,:]
        x_e_h_p = x[:,:,[0,1,3],:,:]
        x_e_h = x[:,:,:2,:,:]
        x2 = self.conv3x3x3(x_e_h_n)
        x3 = self.conv3x5x5(x_e_h_n)
        x4 = self.conv3x7x7(x_e_h_n)
        x5 = self.conv3x3x3(x_e_h_p)
        x6 = self.conv3x5x5(x_e_h_p)
        x7 = self.conv3x7x7(x_e_h_p)
        x8 = self.conv2x3x3(x_e_h)
        x9 = self.conv2x5x5(x_e_h)
        x10 = self.conv2x7x7(x_e_h)
        x_p = torch.cat((x5,x6,x7),dim=1).view(-1,6,56,56)
        x_n = torch.cat((x2,x3,x4),dim=1).view(-1,6,56,56)
        x_neu = torch.cat((x8,x9,x10),dim=1).view(-1,6,56,56)
        x_p=self.encoder(x_p)
        x_p=self.decoder(x_p)
        # print(x_p.shape)
        x_n=self.encoder(x_n)
        x_n=self.decoder(x_n)
        # print(x_n.shape)
        x_neu=self.encoder(x_neu)
        x_neu=self.decoder(x_neu)
        x_neu = x_neu - x_p - x_n
        # print(x_neu.shape)
        output = self.fc(torch.cat((x_p,x_n,x_neu),dim=1))
        # print(output.shape)

        return output


class CNN3D_withpool_90k(nn.Module):
    def __init__(self):
        super(CNN3D_withpool_90k,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
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
UNet_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 56, 56]             592
       BatchNorm2d-2           [-1, 16, 56, 56]              32
           Dropout-3           [-1, 16, 56, 56]               0
         LeakyReLU-4           [-1, 16, 56, 56]               0
         Conv_UNet-5           [-1, 16, 56, 56]               0
            Conv2d-6           [-1, 32, 28, 28]           2,080
         LeakyReLU-7           [-1, 32, 28, 28]               0
 DownSampling_UNet-8           [-1, 32, 28, 28]               0
            Conv2d-9           [-1, 32, 28, 28]           9,248
      BatchNorm2d-10           [-1, 32, 28, 28]              64
          Dropout-11           [-1, 32, 28, 28]               0
        LeakyReLU-12           [-1, 32, 28, 28]               0
        Conv_UNet-13           [-1, 32, 28, 28]               0
           Conv2d-14           [-1, 64, 14, 14]           8,256
        LeakyReLU-15           [-1, 64, 14, 14]               0
DownSampling_UNet-16           [-1, 64, 14, 14]               0
           Conv2d-17           [-1, 64, 14, 14]          36,928
      BatchNorm2d-18           [-1, 64, 14, 14]             128
          Dropout-19           [-1, 64, 14, 14]               0
        LeakyReLU-20           [-1, 64, 14, 14]               0
        Conv_UNet-21           [-1, 64, 14, 14]               0
           Conv2d-22           [-1, 32, 28, 28]           2,080
  UpSampling_UNet-23           [-1, 64, 28, 28]               0
           Conv2d-24           [-1, 32, 28, 28]          18,464
      BatchNorm2d-25           [-1, 32, 28, 28]              64
          Dropout-26           [-1, 32, 28, 28]               0
        LeakyReLU-27           [-1, 32, 28, 28]               0
        Conv_UNet-28           [-1, 32, 28, 28]               0
           Conv2d-29           [-1, 16, 56, 56]             528
  UpSampling_UNet-30           [-1, 32, 56, 56]               0
           Conv2d-31            [-1, 8, 56, 56]           2,312
      BatchNorm2d-32            [-1, 8, 56, 56]              16
          Dropout-33            [-1, 8, 56, 56]               0
        LeakyReLU-34            [-1, 8, 56, 56]               0
        Conv_UNet-35            [-1, 8, 56, 56]               0
           Conv2d-36            [-1, 1, 56, 56]              73
          Sigmoid-37            [-1, 1, 56, 56]               0
================================================================
Total params: 80,865
Trainable params: 80,865
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 7.90
Params size (MB): 0.31
Estimated Total Size (MB): 8.25
----------------------------------------------------------------
'''''''''''

class UNet_90k(nn.Module):
    def __init__(self,in_channels):
        super(UNet_90k, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 16)
        self.D1 = DownSampling_UNet(16, 32)
        self.C2 = Conv_UNet(32, 32)
        self.D2 = DownSampling_UNet(32, 64)
        self.C3 = Conv_UNet(64, 64)
        self.U1 = UpSampling_UNet(64, 32)
        self.C4 = Conv_UNet(64, 32)
        self.U2 = UpSampling_UNet(32, 16)
        self.C5 = Conv_UNet(32, 8)
        self.pred = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        up1 = self.C4(self.U1(R3, R2))
        c = self.C5(self.U2(up1,R1))
        return self.sigmoid(self.pred(c))

'''''''''''
UnetwithSEBlock_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 56, 56]             592
       BatchNorm2d-2           [-1, 16, 56, 56]              32
           Dropout-3           [-1, 16, 56, 56]               0
         LeakyReLU-4           [-1, 16, 56, 56]               0
         Conv_UNet-5           [-1, 16, 56, 56]               0
            Conv2d-6           [-1, 32, 28, 28]           2,080
         LeakyReLU-7           [-1, 32, 28, 28]               0
 DownSampling_UNet-8           [-1, 32, 28, 28]               0
            Conv2d-9           [-1, 32, 28, 28]           9,248
      BatchNorm2d-10           [-1, 32, 28, 28]              64
          Dropout-11           [-1, 32, 28, 28]               0
        LeakyReLU-12           [-1, 32, 28, 28]               0
        Conv_UNet-13           [-1, 32, 28, 28]               0
           Conv2d-14           [-1, 64, 14, 14]           8,256
        LeakyReLU-15           [-1, 64, 14, 14]               0
DownSampling_UNet-16           [-1, 64, 14, 14]               0
           Conv2d-17           [-1, 64, 14, 14]          36,928
      BatchNorm2d-18           [-1, 64, 14, 14]             128
          Dropout-19           [-1, 64, 14, 14]               0
        LeakyReLU-20           [-1, 64, 14, 14]               0
        Conv_UNet-21           [-1, 64, 14, 14]               0
           Linear-22                    [-1, 4]             256
           Linear-23                   [-1, 64]             256
          SEBlock-24           [-1, 64, 14, 14]               0
           Conv2d-25           [-1, 32, 28, 28]           2,080
  UpSampling_UNet-26           [-1, 64, 28, 28]               0
           Conv2d-27           [-1, 32, 28, 28]          18,464
      BatchNorm2d-28           [-1, 32, 28, 28]              64
          Dropout-29           [-1, 32, 28, 28]               0
        LeakyReLU-30           [-1, 32, 28, 28]               0
        Conv_UNet-31           [-1, 32, 28, 28]               0
           Conv2d-32           [-1, 16, 56, 56]             528
  UpSampling_UNet-33           [-1, 32, 56, 56]               0
           Conv2d-34            [-1, 8, 56, 56]           2,312
      BatchNorm2d-35            [-1, 8, 56, 56]              16
          Dropout-36            [-1, 8, 56, 56]               0
        LeakyReLU-37            [-1, 8, 56, 56]               0
        Conv_UNet-38            [-1, 8, 56, 56]               0
           Conv2d-39            [-1, 1, 56, 56]              73
          Sigmoid-40            [-1, 1, 56, 56]               0
================================================================
Total params: 81,377
Trainable params: 81,377
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 7.99
Params size (MB): 0.31
Estimated Total Size (MB): 8.35
----------------------------------------------------------------
'''''''''''
class UnetwithSEBlock_90k(nn.Module):
    def __init__(self,in_channels):
        super(UnetwithSEBlock_90k, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 16)
        self.D1 = DownSampling_UNet(16, 32)
        self.C2 = Conv_UNet(32, 32)
        self.D2 = DownSampling_UNet(32, 64)
        self.C3 = Conv_UNet(64, 64)
        self.se1=SEBlock(64)
        self.U1 = UpSampling_UNet(64, 32)
        self.C4 = Conv_UNet(64, 32)
        self.U2 = UpSampling_UNet(32, 16)
        self.C5 = Conv_UNet(32, 8)
        self.pred = nn.Conv2d(8, 1, kernel_size=3, padding=1)
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
UnetwithSelfattention_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 56, 56]             592
       BatchNorm2d-2           [-1, 16, 56, 56]              32
           Dropout-3           [-1, 16, 56, 56]               0
         LeakyReLU-4           [-1, 16, 56, 56]               0
         Conv_UNet-5           [-1, 16, 56, 56]               0
            Conv2d-6           [-1, 32, 28, 28]           2,080
         LeakyReLU-7           [-1, 32, 28, 28]               0
 DownSampling_UNet-8           [-1, 32, 28, 28]               0
            Conv2d-9           [-1, 32, 28, 28]           9,248
      BatchNorm2d-10           [-1, 32, 28, 28]              64
          Dropout-11           [-1, 32, 28, 28]               0
        LeakyReLU-12           [-1, 32, 28, 28]               0
        Conv_UNet-13           [-1, 32, 28, 28]               0
           Conv2d-14           [-1, 64, 14, 14]           8,256
        LeakyReLU-15           [-1, 64, 14, 14]               0
DownSampling_UNet-16           [-1, 64, 14, 14]               0
           Conv2d-17           [-1, 64, 14, 14]          36,928
      BatchNorm2d-18           [-1, 64, 14, 14]             128
          Dropout-19           [-1, 64, 14, 14]               0
        LeakyReLU-20           [-1, 64, 14, 14]               0
        Conv_UNet-21           [-1, 64, 14, 14]               0
           Conv2d-22           [-1, 64, 14, 14]           4,160
           Conv2d-23           [-1, 64, 14, 14]           4,160
           Conv2d-24           [-1, 64, 14, 14]           4,160
          Softmax-25             [-1, 196, 196]               0
    SelfAttention-26           [-1, 64, 14, 14]               0
           Conv2d-27           [-1, 32, 28, 28]           2,080
  UpSampling_UNet-28           [-1, 64, 28, 28]               0
           Conv2d-29           [-1, 32, 28, 28]          18,464
      BatchNorm2d-30           [-1, 32, 28, 28]              64
          Dropout-31           [-1, 32, 28, 28]               0
        LeakyReLU-32           [-1, 32, 28, 28]               0
        Conv_UNet-33           [-1, 32, 28, 28]               0
           Conv2d-34           [-1, 16, 56, 56]             528
  UpSampling_UNet-35           [-1, 32, 56, 56]               0
           Conv2d-36            [-1, 8, 56, 56]           2,312
      BatchNorm2d-37            [-1, 8, 56, 56]              16
          Dropout-38            [-1, 8, 56, 56]               0
        LeakyReLU-39            [-1, 8, 56, 56]               0
        Conv_UNet-40            [-1, 8, 56, 56]               0
           Conv2d-41            [-1, 1, 56, 56]              73
          Sigmoid-42            [-1, 1, 56, 56]               0
================================================================
Total params: 93,345
Trainable params: 93,345
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 8.57
Params size (MB): 0.36
Estimated Total Size (MB): 8.98
----------------------------------------------------------------
'''''''''''''''
class UnetwithSelfattention_90k(nn.Module):
    def __init__(self,in_channels):
        super(UnetwithSelfattention_90k, self).__init__()
        self.in_channels=in_channels
        self.C1 = Conv_UNet(self.in_channels, 16) 
        self.D1 = DownSampling_UNet(16, 32) 
        self.C2 = Conv_UNet(32, 32)
        self.D2 = DownSampling_UNet(32, 64) 
        self.C3 = Conv_UNet(64, 64)
        self.attention = SelfAttention(64)
        self.U1 = UpSampling_UNet(64, 32) 
        self.C4 = Conv_UNet(64, 32)  
        self.U2 = UpSampling_UNet(32, 16) 
        self.C5 = Conv_UNet(32, 8)  
        self.pred = nn.Conv2d(8, 1, kernel_size=3, padding=1)
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
Unet3D_90k:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1         [-1, 2, 1, 56, 56]              56
            Conv3d-2         [-1, 2, 1, 56, 56]             152
            Conv3d-3         [-1, 2, 1, 56, 56]             296
            Conv3d-4         [-1, 2, 1, 56, 56]              56
            Conv3d-5         [-1, 2, 1, 56, 56]             152
            Conv3d-6         [-1, 2, 1, 56, 56]             296
            Conv2d-7           [-1, 16, 56, 56]           1,744
       BatchNorm2d-8           [-1, 16, 56, 56]              32
           Dropout-9           [-1, 16, 56, 56]               0
        LeakyReLU-10           [-1, 16, 56, 56]               0
        Conv_UNet-11           [-1, 16, 56, 56]               0
           Conv2d-12           [-1, 32, 28, 28]           2,080
        LeakyReLU-13           [-1, 32, 28, 28]               0
DownSampling_UNet-14           [-1, 32, 28, 28]               0
           Conv2d-15           [-1, 32, 28, 28]           9,248
      BatchNorm2d-16           [-1, 32, 28, 28]              64
          Dropout-17           [-1, 32, 28, 28]               0
        LeakyReLU-18           [-1, 32, 28, 28]               0
        Conv_UNet-19           [-1, 32, 28, 28]               0
           Conv2d-20           [-1, 64, 14, 14]           8,256
        LeakyReLU-21           [-1, 64, 14, 14]               0
DownSampling_UNet-22           [-1, 64, 14, 14]               0
           Conv2d-23           [-1, 64, 14, 14]          36,928
      BatchNorm2d-24           [-1, 64, 14, 14]             128
          Dropout-25           [-1, 64, 14, 14]               0
        LeakyReLU-26           [-1, 64, 14, 14]               0
        Conv_UNet-27           [-1, 64, 14, 14]               0
           Conv2d-28           [-1, 32, 28, 28]           2,080
  UpSampling_UNet-29           [-1, 64, 28, 28]               0
           Conv2d-30           [-1, 32, 28, 28]          18,464
      BatchNorm2d-31           [-1, 32, 28, 28]              64
          Dropout-32           [-1, 32, 28, 28]               0
        LeakyReLU-33           [-1, 32, 28, 28]               0
        Conv_UNet-34           [-1, 32, 28, 28]               0
           Conv2d-35           [-1, 16, 56, 56]             528
  UpSampling_UNet-36           [-1, 32, 56, 56]               0
           Conv2d-37            [-1, 8, 56, 56]           2,312
      BatchNorm2d-38            [-1, 8, 56, 56]              16
          Dropout-39            [-1, 8, 56, 56]               0
        LeakyReLU-40            [-1, 8, 56, 56]               0
        Conv_UNet-41            [-1, 8, 56, 56]               0
           Conv2d-42            [-1, 1, 56, 56]              73
          Sigmoid-43            [-1, 1, 56, 56]               0
             UNet-44            [-1, 1, 56, 56]               0
================================================================
Total params: 83,025
Trainable params: 83,025
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 8.21
Params size (MB): 0.32
Estimated Total Size (MB): 8.57
----------------------------------------------------------------
'''''''''''''''''

class Unet3D_90k(nn.Module):
    def __init__(self):
        super(Unet3D_90k,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1)) #(batch_size,2,1,56,56)
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2)) #(batch_size,2,1,56,56)
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3)) #(batch_size,2,1,56,56)
        self.unet=UNet_90k(12)
    
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

