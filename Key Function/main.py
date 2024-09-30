import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset, random_split
import tifffile as tiff #load tiff image
from PIL import Image
import os
import torch
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import imageio

WIDTH = 56
HEIGHT = 56
#定义超参数

#导入数据
# #from sklearn.model_selection import train_test_split
# 对于图片我们要转化为tensor，先定义一个Transform
transform=transforms.Compose([
    transforms.ToTensor(),
])
# 注意：这里的路径要自己修改
directory="Gauss_S1.00_NL0.30_B0.00"
#我这里采用的方法是用for循环把所有照片都读进来再分类
all_imgs=[]
for filename in os.listdir(directory):
    if(filename.endswith(".tiff")):
        img_path=os.path.join(directory, filename)
    img_array=tiff.imread(img_path)
    img=Image.fromarray(img_array)
    img_tensor=transform(img)
    all_imgs.append(img_tensor)
print(f'我已经读取{len(all_imgs)}张图像')
#for idx, tensor in enumerate(all_imgs):
    #print(f'图像 {idx+1} 的张量形状:{tensor.shape}')
#把all_imgs分成4类
Emcal=[]
Hcal=[]
Tracker=[]
Trkp=[]
Truth=[]
#p->per_imgs
p=10000
Emcal=all_imgs[:p]
Hcal=all_imgs[p:2*p]
Tracker=all_imgs[2*p:3*p]
Trkp=all_imgs[3*p:4*p]
Truth=all_imgs[4*p:5*p]
#我把三张子图合并起来（相当于三个通道）
X=[]
for emcal, hcal, tracker,trkp in zip(Emcal, Hcal, Tracker,Trkp):
    combined_features=torch.stack((emcal, hcal, tracker,trkp))
    X.append(combined_features)
X=torch.stack(X)
X=X.squeeze(2)
Y=torch.stack(Truth)
print(f'X的尺寸是:{X.shape}')
print(f'Y的尺寸是:{Y.shape}')

dataset_1=torch.cat((X,Y),dim=1)
print(dataset_1.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import random

 #随机抽取50张图片出来做测试集

all_picture=dataset_1.numpy()#张量是静态，需要先转换为numpy数组，再进行抽取和删减操作
print(all_picture.shape)

#抽取策略：分层抽样(比如：每20取1)
random_number = random.randint(1, 20)#随机生成一个1~20的随机数
print(random_number)
index=[]
for i in range(50):
    index.append(20*i+random_number-1)
print(index)

#这里先提取索引对应图片数据
test_picture50=all_picture[index]
print(test_picture50.shape)
#随后删除——将不再index中的索引图片copy到my_pictures中
my_pictures = np.delete(all_picture,index,axis=0)
print(my_pictures.shape)

#转换为tensor
dataset_1_1=torch.tensor(my_pictures)
print(dataset_1_1.shape)

#转换形状
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
# 可选参数还有requires_grad， 这是能否自动求梯度的关键，单独写的时候是关的
# dtype也很重要，因为cuda一般只支持float32
# print(my_tensor.shape)
# print(my_tensor.reshape(6))
dataset_2=dataset_1_1.permute(1,0,2,3);
print(dataset_2.shape)
dataset_3=dataset_2.reshape(5,31203200)
print(dataset_3.shape)
dataset_3=dataset_3.T
print(dataset_3.shape)
dataset_3

#对数变换
dataset_3_ln=torch.log(dataset_3+1)
dataset_3_ln

#归一化
# 沿着适当的维度找到最小值和最大值,dim等于0（在竖直方向上）
min_values = torch.min(dataset_3_ln, dim=0).values
max_values = torch.max(dataset_3_ln, dim=0).values
print(min_values)
print(max_values)
# 对张量进行最小-最大归一化
normalized_dataset = (dataset_3_ln - min_values) / (max_values - min_values)
normalized_dataset=normalized_dataset.squeeze(0)
# 打印归一化后的张量
print(normalized_dataset.shape)

#合成集（分开x，y）
X=normalized_dataset[:70000,:4]
print(X.shape)
X_1=X.squeeze(0)
print(X_1.shape)
Y=normalized_dataset[:70000,4]
print(Y.shape)
dataset_all=TensorDataset(X_1, Y)
print(dataset_all)

#mlp训练模型
#分割数据集
#TRAIN_NUM=2200000
#VAL_NUM=779200

TRAIN_NUM=50000
VAL_NUM=10000
TEST_NUM=10000
train_dataset, val_dataset, test_dataset = data.random_split(
    dataset_all, [TRAIN_NUM, VAL_NUM,TEST_NUM]
    )
print(list(train_dataset))
print(list(val_dataset))
print(list(test_dataset))

# Data loaders
BATCH_SIZE = 2000
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) #test_data待处理


# Define a simple fully connected network
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)  # Output size matches the flattened Y

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
NUM_EPOCHS = 15
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Adjust target tensor size
        Y_batch = Y_batch.unsqueeze(1)  # 将目标张量调整为 (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            # Adjust target tensor size
            Y_val = Y_val.unsqueeze(1)  # 将目标张量调整为 (batch_size, 1)

            outputs = model(X_val)
            val_loss += criterion(outputs, Y_val).item()
        val_loss /= len(val_loader)

    print(
        f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')

print('Training completed!')

# Optional: Evaluate on the test set
model.eval()
test_loss = 0
with torch.no_grad():
    for X_test, Y_test in test_loader:
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        outputs = model(X_test)
        test_loss += criterion(outputs, Y_test).item()
    test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

print(test_picture50.shape)

one_of_picture=test_picture50[7,:4,:,:]
print(one_of_picture.shape)
aim_of_truth=test_picture50[7,4,:,:]
print(aim_of_truth.shape)

X_pic=torch.tensor(one_of_picture)
print(X_pic.shape)
X_pic=X_pic.reshape(4,3136)
X_pic=X_pic.T
print(X_pic)
X_pic=X_pic.squeeze(0)
print(X_pic)

Y_outputs = model(X_pic)
print(Y_outputs.shape)

v=Y_outputs.reshape(56,56)
v_array=v.detach().numpy()
print(v_array)
# 绘制灰度图
import matplotlib.pyplot as plt


# 创建一个 1x2 的子图布局
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 绘制第一个灰度图aim_of_truth(56*56)
axes[0].imshow(aim_of_truth,cmap="Greens")
axes[0].set_title('Truth Image')
axes[0].axis('off')  # 隐藏坐标轴

# 绘制第二个灰度图v_array(56*56)
axes[1].imshow(v_array,cmap="Greens")
axes[1].set_title('Predict Image')
axes[1].axis('off')  # 隐藏坐标轴

# 显示图像
plt.show()