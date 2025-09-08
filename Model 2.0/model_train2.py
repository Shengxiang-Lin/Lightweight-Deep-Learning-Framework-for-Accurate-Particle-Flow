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
        



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x


class CNNwithSEBlock(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.se1=SEBlock(512)
        self.decoder=nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.se1(x)
        x=self.decoder(x)
        return x

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D,self).__init__()
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
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, padding=2),
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


class CNNwithSEBlock3D(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock3D,self).__init__()
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
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.se1=SEBlock(256)
        self.decoder=nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=5, padding=2),
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
        x=self.se1(x)
        x=self.decoder(x)
        return x
    


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.C1 = Conv_UNet(4, 128)
        self.D1 = DownSampling_UNet(128, 256)
        self.C2 = Conv_UNet(256, 256)
        self.D2 = DownSampling_UNet(256, 512)
        self.C3 = Conv_UNet(512, 512)
        self.D3 = DownSampling_UNet(512, 1024)
        self.C4 = Conv_UNet(1024, 1024)
        self.U1 = UpSampling_UNet(1024, 512)
        self.C5 = Conv_UNet(1024, 512)
        self.U2 = UpSampling_UNet(512, 256)
        self.C6 = Conv_UNet(512, 256)
        self.U3 = UpSampling_UNet(256, 128)
        self.C7 = Conv_UNet(256, 128)
        self.pred = nn.Conv2d(128, 1, kernel_size=3, padding=1)
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



class UNetwithSEBlock(nn.Module):
    def __init__(self):
        super(UNetwithSEBlock, self).__init__()
        self.C1 = Conv_UNet(4, 128)
        self.D1 = DownSampling_UNet(128, 256)
        self.C2 = Conv_UNet(256, 256)
        self.D2 = DownSampling_UNet(256, 512)
        self.C3 = Conv_UNet(512, 512)
        self.D3 = DownSampling_UNet(512, 1024)
        self.C4 = Conv_UNet(1024, 1024)
        self.se1=SEBlock(1024)
        self.U1 = UpSampling_UNet(1024, 512)
        self.C5 = Conv_UNet(1024, 512)
        self.U2 = UpSampling_UNet(512, 256)
        self.C6 = Conv_UNet(512, 256)
        self.U3 = UpSampling_UNet(256, 128)
        self.C7 = Conv_UNet(256, 128)
        self.pred = nn.Conv2d(128, 1, kernel_size=3, padding=1)
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



class UNetwithSelfattention(nn.Module):
    def __init__(self):
        super(UNetwithSelfattention, self).__init__()
        self.C1 = Conv_UNet(4, 128)
        self.D1 = DownSampling_UNet(128, 256)
        self.C2 = Conv_UNet(256, 256)
        self.D2 = DownSampling_UNet(256, 512)
        self.C3 = Conv_UNet(512, 512)
        self.D3 = DownSampling_UNet(512, 1024)
        self.C4 = Conv_UNet(1024, 1024)
        self.attention = SelfAttention(1024)
        self.U1 = UpSampling_UNet(1024, 512)
        self.C5 = Conv_UNet(1024, 512)
        self.U2 = UpSampling_UNet(512, 256)
        self.C6 = Conv_UNet(512, 256)
        self.U3 = UpSampling_UNet(256, 128)
        self.C7 = Conv_UNet(256, 128)
        self.pred = nn.Conv2d(128, 1, kernel_size=3, padding=1)
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



class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1)) #(batch_size,2,1,56,56)
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2)) #(batch_size,2,1,56,56)
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3)) #(batch_size,2,1,56,56)
        self.C1 = Conv_UNet(12, 128)
        self.D1 = DownSampling_UNet(128, 256)
        self.C2 = Conv_UNet(256, 256)
        self.D2 = DownSampling_UNet(256, 512)
        self.C3 = Conv_UNet(512, 512)
        self.D3 = DownSampling_UNet(512, 1024)
        self.C4 = Conv_UNet(1024, 1024)
        self.U1 = UpSampling_UNet(1024, 512)
        self.C5 = Conv_UNet(1024, 512)
        self.U2 = UpSampling_UNet(512, 256)
        self.C6 = Conv_UNet(512, 256)
        self.U3 = UpSampling_UNet(256, 128)
        self.C7 = Conv_UNet(256, 128)
        self.pred = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
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
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))
        



class UNetwithSEBlock3D(nn.Module):
    def __init__(self):
        super(UNetwithSEBlock3D, self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.C1 = Conv_UNet(12, 128)
        self.D1 = DownSampling_UNet(128, 256)
        self.C2 = Conv_UNet(256, 256)
        self.D2 = DownSampling_UNet(256, 512)
        self.C3 = Conv_UNet(512, 512)
        self.D3 = DownSampling_UNet(512, 1024)
        self.C4 = Conv_UNet(1024, 1024)
        self.se1=SEBlock(1024)
        self.U1 = UpSampling_UNet(1024, 512)
        self.C5 = Conv_UNet(1024, 512)
        self.U2 = UpSampling_UNet(512, 256)
        self.C6 = Conv_UNet(512, 256)
        self.U3 = UpSampling_UNet(256, 128)
        self.C7 = Conv_UNet(256, 128)
        self.pred = nn.Conv2d(128, 1, kernel_size=3, padding=1)
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
        x = torch.cat((x2,x3,x4,x5,x6,x7),dim=1).view(-1,12,56,56)
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        R4=self.se1(R4)
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))



class UNetwithSelfattention3D(nn.Module):
    def __init__(self):
        super(UNetwithSelfattention3D, self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.C1 = Conv_UNet(12, 128)
        self.D1 = DownSampling_UNet(128, 256)
        self.C2 = Conv_UNet(256, 256)
        self.D2 = DownSampling_UNet(256, 512)
        self.C3 = Conv_UNet(512, 512)
        self.D3 = DownSampling_UNet(512, 1024)
        self.C4 = Conv_UNet(1024, 1024)
        self.attention = SelfAttention(1024)
        self.U1 = UpSampling_UNet(1024, 512)
        self.C5 = Conv_UNet(1024, 512)
        self.U2 = UpSampling_UNet(512, 256)
        self.C6 = Conv_UNet(512, 256)
        self.U3 = UpSampling_UNet(256, 128)
        self.C7 = Conv_UNet(256, 128)
        self.pred = nn.Conv2d(128, 1, kernel_size=3, padding=1)
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
        x = torch.cat((x2,x3,x4,x5,x6,x7),dim=1).view(-1,12,56,56)
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        R4=self.attention(R4)
        up1 = self.C5(self.U1(R4, R3))
        up2 = self.C6(self.U2(up1, R2))
        up3 = self.C7(self.U3(up2, R1))
        return self.sigmoid(self.pred(up3))
