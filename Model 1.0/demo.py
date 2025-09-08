'''''''''''
统计模型参数量
'''''''''''

# import torch
# from torchsummary import summary
# from model_90k import CNN3Dver3_90k


# model = CNN3Dver3_90k().to("cuda")
# print(summary(model, (4, 56, 56)))

'''''''''
计算预测结果偏差以及重心偏移
'''''''''
# import numpy as np
# import torch
# import tifffile
# import os
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import shutil
# from DataSet import MaxMinNormalizeGlobalPerChannel,MyDataSet, dataset_2
# from model_5m import UNet_5m
# import sys

# data_transform = {
# 	"without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()]),
# 	"jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()])}
# data_set = MyDataSet(img_dir='Gauss_S1.00_NL0.30_B0.50\Gauss_S1.00_NL0.30_B0.50',
# 								group_size=10000,
# 								size_in = 200,
# 								splition= False,
# 								split_shuffle = False,
# 								transform=data_transform["without_jet"])
# X=data_set.X #(10000,4,56,56)
# Y=data_set.Y #(10000,1,56,56)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# jet_dataset=dataset_2(X,Y)
# #分割数据集
# model = UNet_5m(in_channels=4).to(device)
# model.load_state_dict(torch.load('weights\\UNet_5m.pth'))
# TEST_NUM=10000
# BATCH_SIZE=200
# Y_min = 0.0
# Y_max = 20.6504
# print(TEST_NUM)
# test_loader_jet = DataLoader(jet_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
	
# model.eval()
# predicted_images=[]

# with torch.no_grad():
# 	for i,(X_test, Y_test) in enumerate(test_loader_jet):
# 		outputs=model(X_test.to(device)) * (Y_max - Y_min) + Y_min
# 		predicted_images.append(outputs.cpu().detach().numpy())
# predicted_images = np.concatenate(predicted_images, axis=0)
# print(predicted_images.shape)

# # sys.exit()

# truth_list = Y.numpy()
# predict_list = predicted_images
# energy_residual_0 = []
# energy_residual_0to5 = []
# energy_residual_5to10 = []
# energy_residual_10to15 = []
# energy_residual_15toinfty = []

# barycenter_shift = []
# j = 0
# for true_img, predict_img in zip(truth_list, predict_list):
#     true_img = true_img[0]
#     predict_img = predict_img[0]
#     barycenter_true_X = 0
#     barycenter_true_Y = 0
#     barycenter_predict_X = 0
#     barycenter_predict_Y = 0
#     for i in range(56):
#         for j in range(56):
#             if true_img[i][j] == 0:
#                 energy_residual_0.append(predict_img[i][j])
#             elif true_img[i][j] > 1e-1:
#                 energy_residual = np.abs((true_img[i][j] - predict_img[i][j]) / true_img[i][j])
#                 if true_img[i][j] > 0.1 and true_img[i][j] <= 5:
#                     energy_residual_0to5.append(energy_residual)
#                 elif true_img[i][j] > 5 and true_img[i][j] <= 10:
#                     energy_residual_5to10.append(energy_residual)
#                 elif true_img[i][j] > 10 and true_img[i][j] <= 15:
#                     energy_residual_10to15.append(energy_residual)
#                 elif true_img[i][j] > 15:
#                     energy_residual_15toinfty.append(energy_residual)
#             barycenter_true_X += i * true_img[i][j]
#             barycenter_true_Y += j * true_img[i][j]
#             barycenter_predict_X += i * predict_img[i][j]
#             barycenter_predict_Y += j * predict_img[i][j]
#     barycenter_true_X = barycenter_true_X / true_img.sum()
#     barycenter_true_Y = barycenter_true_Y / true_img.sum()
#     barycenter_predict_X = barycenter_predict_X / predict_img.sum()
#     barycenter_predict_Y = barycenter_predict_Y / predict_img.sum()
#     barycenter_shift.append(np.sqrt((barycenter_true_X - barycenter_predict_X) ** 2 + (barycenter_true_Y - barycenter_predict_Y) ** 2))
#     j += 1
#     if j % 1000 == 0:
#         print(j)
# def compute_rms(values):
#     arr = np.array(values)
#     return np.sqrt(np.nanmean(arr ** 2)) 

# def compute_variance(values):
#     arr = np.array(values)
#     return np.nanvar(arr)  

# def compute_mean(values):
#     arr = np.array(values)
#     return np.nanmean(arr)
# # metrics = {
# #     "energy_residual_0": energy_residual_0,
# #     "energy_residual_0to5": energy_residual_0to5,
# #     "energy_residual_5to10": energy_residual_5to10,
# #     "energy_residual_10to15": energy_residual_10to15,
# #     "energy_residual_15toinfty": energy_residual_15toinfty,
# #     "barycenter_shift": barycenter_shift
# # }
# metrics = {
#     "barycenter_shift": barycenter_shift
# }

# for key, values in metrics.items():
#     rms = compute_rms(values)
#     mean = compute_mean(values)
#     print(f"{key}: MEAN = {rms:.4f}, 方差 = {mean:.4f}")

'''''''''''
将预测结果转换为jet
'''''''''''

# import os
# import tifffile as tiff
# import sys
# import math
# directory_path = 'predicted_images\\CNN3Dver2_90k'
# sys.stdout=open('predicted_images\\output_CNN3Dver2_90k_predicted.txt','w')
# images = []
# nums = []
# for num in range(1000):
#     filename = f"predict_{str(num)}.tiff"
#     file_path = os.path.join(directory_path, filename)
#     image = tiff.imread(file_path)
#     images.append(image)
#     num=0
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel_value = image[i, j]
#             if pixel_value>0:
#                 num=num+1
#     nums.append(num)
# for i in range(len(nums)):
#     print(nums[i])
# for num in range(1000):
#     filename = f"predict_{str(num)}.tiff"
#     file_path = os.path.join(directory_path, filename)
#     image = tiff.imread(file_path)
#     images.append(image)
#     for i in range(image.shape[0]): 
#         for j in range(image.shape[1]):
#             pixel_value = image[i, j]
#             if pixel_value>0:
#                 pz=i+0.5-image.shape[0]/2
#                 py=(56/2/math.pi)*math.cos((j+0.5)/56*2*math.pi)
#                 px=(56/2/math.pi)*math.sin((j+0.5)/56*2*math.pi)
#                 pr=math.sqrt(px*px+py*py+pz*pz)
#                 px=px/pr
#                 py=py/pr
#                 pz=pz/pr
#                 print(f'{px} {py} {pz} {pixel_value}') 


'''''''''''
绘制模型在jet数据集上的预测结果
'''''''''''

# import torch
# import tifffile
# import os
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import math

# from DataSet import MaxMinNormalizeGlobalPerChannel,MyDataSet, dataset_2
# from model_90k import CNN3D_90k


# ''''''''''''
# #假设已有模型,没有加载好jet数据集
# ''''''''''''
# data_transform = {
# 	"without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()]),
# 	"jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()])}
# data_set_jet = MyDataSet(img_dir='Gauss_S1.00_NL0.30_B0.50_Jet\Gauss_S1.00_NL0.30_B0.50_Jet',
# 								group_size=1000,
# 								size_in = 1000,
# 								splition= False,
# 								split_shuffle = False,
# 								transform=data_transform["jet"])
# X=data_set_jet.X #(10000,4,56,56)
# Y=data_set_jet.Y #(10000,1,56,56)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# jet_dataset=dataset_2(X,Y)
# #分割数据集
# model = CNN3D_90k().to(device)
# model.load_state_dict(torch.load('weights\\CNN3D_90k.pth'))
# TEST_NUM=1000
# BATCH_SIZE=200
# print(TEST_NUM)
# test_loader_jet = DataLoader(jet_dataset, batch_size=BATCH_SIZE, shuffle=False)
	
# model.eval()
# predicted_images=[]

# with torch.no_grad():
# 	for i,(X_test, Y_test) in enumerate(test_loader_jet):
# 		outputs=model(X_test.to(device))
# 		predicted_images.append(outputs.cpu().detach().numpy())
# predicted_images = np.concatenate(predicted_images, axis=0)
# print(predicted_images.shape)
# # sys.exit()
# sys.stdout=open('predicted_images\\output_CNN3D_90k_predict.txt','w')
# nums = []
# for image in predicted_images:
#     image = image[0]
#     num=0
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel_value = image[i, j]
#             if pixel_value>0:
#                 num=num+1
#     nums.append(num)
# for i in range(len(nums)):
#     print(nums[i])
# for image in predicted_images:
#     image = image[0]
#     for i in range(image.shape[0]): 
#         for j in range(image.shape[1]):
#             pixel_value = image[i, j]
#             if pixel_value>0:
#                 pz=i+0.5-image.shape[0]/2
#                 py=(56/2/math.pi)*math.cos((j+0.5)/56*2*math.pi)
#                 px=(56/2/math.pi)*math.sin((j+0.5)/56*2*math.pi)
#                 pr=math.sqrt(px*px+py*py+pz*pz)
#                 px=px/pr
#                 py=py/pr
#                 pz=pz/pr
#                 print(f'{px} {py} {pz} {pixel_value}') 
# sys.stdout.close()

'''''''''''''''
绘制学习率曲线
'''''''''''''''


# import math
# import matplotlib.pyplot as plt

# warmup_epochs_1 = 40
# warmup_epochs_2 = 43
# warmup_epochs_3 = 60
# warmup_epochs_4 = 80
# warmup_epochs_5 = 83
# epochs = 100
# lr = 0.001

# lr_string = []

# def lf(x): 
#     if x < warmup_epochs_1:
#         return(lr)
#     elif x < warmup_epochs_2:
#         return((x - warmup_epochs_1) / (warmup_epochs_2 - warmup_epochs_1)) * (lr * 10 - lr) + lr
#     elif x < warmup_epochs_3: 
#         return(((1 + math.cos((x - warmup_epochs_2) * math.pi / (warmup_epochs_3 - warmup_epochs_2))) / 2) * (lr * 10 - lr / 10) + lr / 10)
#     elif x < warmup_epochs_4: 
#         return(lr / 10)
#     elif x < warmup_epochs_5:
#         return((x - warmup_epochs_4) / (warmup_epochs_5 - warmup_epochs_4)) * (lr - lr / 10) + lr / 10
#     else:
#         return(((1 + math.cos((x - warmup_epochs_5) * math.pi / (epochs - warmup_epochs_5))) / 2) * (lr - lr / 10) + lr / 10)
        
# for epoch in range(epochs):
#     lr_string.append(lf(epoch))
#     print(f"Epoch {epoch}: Learning rate: {lf(epoch)}")

# plt.plot(lr_string)
# # plt.yscale('log')
# plt.show()

'''''''''
简单检查随机数种子
'''''''''
# import torch
# torch.manual_seed(26)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(26)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# a = torch.randn(10, 3).cuda()
# print(a)


'''''''''''
观察各模型在jet数据集上的预测结果
'''''''''''
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the function to calculate the average of absolute values from a txt file
# def calculate_avg_abs_value(file_path):
#     with open(file_path, 'r') as file:
#         numbers = [float(line.strip()) for line in file]
#     return sum(abs(num) for num in numbers) / len(numbers)

# # Define the folder path containing the files
# folder_path = 'errors\\errors'

# # Initialize variables to store the results
# pts_avg_abs_value = {}
# dis_avg_abs_value = {}

# # Loop through files in the folder
# for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
#     if file_name.endswith('pts.txt'):
#         pts_avg_abs_value[file_name] = calculate_avg_abs_value(file_path)
#     elif file_name.endswith('dis.txt'):
#         dis_avg_abs_value[file_name] = calculate_avg_abs_value(file_path)
    
# plt.figure(figsize=(10, 10))


# # 按照值进行升序排序
# pts_avg_abs_value = sorted(pts_avg_abs_value.items(), key=lambda item: item[1])
# dis_avg_abs_value = sorted(dis_avg_abs_value.items(), key=lambda item: item[1])

# # 提取排序后的键和值
# pts_avg_abs_value_keys = [item[0] for item in pts_avg_abs_value]
# pts_avg_abs_value_values = [item[1] for item in pts_avg_abs_value]
# pts_avg_abs_value_keys = [key.replace('_pts.txt', '') for key in pts_avg_abs_value_keys]

# dis_avg_abs_value_keys = [item[0] for item in dis_avg_abs_value]
# dis_avg_abs_value_values = [item[1] for item in dis_avg_abs_value]
# dis_avg_abs_value_keys = [key.replace('_dis.txt', '') for key in dis_avg_abs_value_keys]

# # 绘制条形图
# # bars = plt.bar(pts_avg_abs_value_keys, pts_avg_abs_value_values, color='blue')
# # bars = plt.bar(dis_avg_abs_value_keys, dis_avg_abs_value_values, color='red')

# # for bar in bars:
# #     yval = bar.get_height()  # 获取条形图的高度，即值
# #     formatted_value = f"{yval:.2f}"  # 格式化值，保留2位小数
# #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5,  # 设置文本位置
# #              formatted_value, ha='center', va='bottom',
# #              rotation = 90)  # 显示格式化后的值


# # # 设置图表标题和标签
# # plt.title("distance between predicted jet and true jet")
# # plt.xlabel("model")
# # plt.ylabel("distance")
# # plt.yscale('log')
# # plt.xticks(rotation=90)
# # plt.tight_layout()
# # # 显示图表
# # # plt.savefig('distance between predicted jet and true jet.svg')
# # plt.show()
# pts_rank = []
# dis_rank = []
# for key in pts_avg_abs_value_keys:
#     pts_rank.append(int(pts_avg_abs_value_keys.index(key)))
#     dis_rank.append(int(dis_avg_abs_value_keys.index(key)))


# bars_1 = plt.bar(pts_avg_abs_value_keys, pts_rank, alpha = 0.6, color = 'r')
# bars_2 = plt.bar(pts_avg_abs_value_keys, dis_rank, alpha = 0.6, color = 'b')

# for key,bar1,bar2 in zip(pts_avg_abs_value_keys,bars_1,bars_2):
#     yval = np.mean([bar1.get_height(), bar2.get_height()])  # 获取条形图的高度，即值
#     y = np.max([bar1.get_height(), bar2.get_height()])
#     formatted_value = f"{yval:.1f}"  # 格式化值，保留2位小数
#     plt.text(bar1.get_x() + bar1.get_width() / 2, y + 0.5,  # 设置文本位置
#              formatted_value, ha='center', va='bottom',
#              rotation = 90)  # 显示格式化后的值
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig('the rank of models.svg')

'''''''''''''''''
使用三个指标评价模型预测结果
'''''''''''''''''

import os
import math
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model_90k import CNN_90k, CNN_withpool_90k,CNNwithSEBlock_90k, CNNwithSEBlock_withpool_90k, CNNwithSelfattention_withpool_90k, CNN3D_90k, CNN3D_withpool_90k, UNet_90k, UnetwithSEBlock_90k, UnetwithSelfattention_90k, Unet3D_90k
# from model_1m import CNN_1m, CNNwithSEBlock_1m, CNNwithSelfattention_1m, CNN3D_1m, UNet_1m, UnetwithSEBlock_1m, UnetwithSelfattention_1m, Unet3D_1m
# from model_5m import CNN_5m, CNNwithSEBlock_5m, CNNwithSelfattention_5m, CNN3D_5m, UNet_5m, UnetwithSEBlock_5m, UnetwithSelfattention_5m, Unet3D_5m
# from model_5m import UNetall_5m
from model_90k import CNN3Dver2_90k
from DataSet import MaxMinNormalizeGlobalPerChannel,MyDataSet, dataset_2

random.seed(26)
np.random.seed(26)
torch.manual_seed(26)
torch.cuda.manual_seed(26)
torch.cuda.manual_seed_all(26) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或者 ":4096:8"

img_dir = 'Gauss_S1.00_NL0.30_B0.50\Gauss_S1.00_NL0.30_B0.50'

model_90k = {
#     "CNN_90k": CNN_90k().to("cuda"),
#     "CNN_withpool_90k": CNN_withpool_90k().to("cuda"),
#     "CNNwithSEBlock_90k": CNNwithSEBlock_90k().to("cuda"),
#     "CNNwithSEBlock_withpool_90k": CNNwithSEBlock_withpool_90k().to("cuda"),
#     "CNNwithSelfattention_withpool_90k": CNNwithSelfattention_withpool_90k().to("cuda"),
#     "CNN3D_90k": CNN3D_90k().to("cuda"),
#     "CNN3D_withpool_90k": CNN3D_withpool_90k().to("cuda"),
#     "UNet_90k": UNet_90k(in_channels=4).to("cuda"),
#     "UnetwithSEBlock_90k": UnetwithSEBlock_90k(in_channels=4).to("cuda"),
#     "UnetwithSelfattention_90k": UnetwithSelfattention_90k(in_channels=4).to("cuda"),
#     "UNet3D_90k": Unet3D_90k().to("cuda")
    "CNN3Dver2_90k": CNN3Dver2_90k().to("cuda")
}

# model_1m = {
#     "CNN_1m": CNN_1m().to("cuda"),
#     "CNNwithSEBlock_1m": CNNwithSEBlock_1m().to("cuda"),
#     "CNNwithSelfattention_1m": CNNwithSelfattention_1m().to("cuda"),
#     "CNN3D_1m": CNN3D_1m().to("cuda"),
#     "UNet_1m": UNet_1m(in_channels=4).to("cuda"),
#     "UnetwithSEBlock_1m": UnetwithSEBlock_1m(in_channels=4).to("cuda"),
#     "UnetwithSelfattention_1m": UnetwithSelfattention_1m(in_channels=4).to("cuda"),
#     "Unet3D_1m": Unet3D_1m().to("cuda")
# }


# model_5m = {
    # "CNN_5m": CNN_5m().to("cuda"),
    # "CNNwithSEBlock_5m": CNNwithSEBlock_5m().to("cuda"),
    # "CNNwithSelfattention_5m": CNNwithSelfattention_5m().to("cuda"),
    # "CNN3D_5m": CNN3D_5m().to("cuda"),
    # "UNet_5m": UNet_5m(in_channels=4).to("cuda"),
    # "UnetwithSEBlock_5m": UnetwithSEBlock_5m(in_channels=4).to("cuda"),
    # "UnetwithSelfattention_5m": UnetwithSelfattention_5m(in_channels=4).to("cuda"),
    # "Unet3D_5m": Unet3D_5m().to("cuda")
    # "UNetall_5m": UNetall_5m().to("cuda")
# }

# 定义训练以及预测时的预处理方法
data_transform = {
    "without_jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()]),
    "jet": transforms.Compose([MaxMinNormalizeGlobalPerChannel()])}

# 实例化训练数据集
data_set = MyDataSet(img_dir=img_dir,
                    group_size=10000,
                    size_in = 10000,
                    splition = True,
                    split_shuffle = False,
                    transform=data_transform["without_jet"])
train_dataset = dataset_2(data_set.train_X, data_set.train_Y)
val_dataset = dataset_2(data_set.val_X, data_set.val_Y)
test_dataset = dataset_2(data_set.test_X, data_set.test_Y)

del train_dataset
del test_dataset

test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=0)

Y_min = 0.0
Y_max = 20.6504

def weight_MSE(predicted_img, true_img):
    weight = true_img / np.sum(true_img, axis=(1, 2), keepdims=True)
    return np.sum(weight * (predicted_img - true_img) ** 2)

class ModelResults:
    def __init__(self):
        self.energy_residual_2to5 = []
        self.energy_residual_5to10 = []
        self.energy_residual_10toinfty = []
        self.barycenter_shift = [[] for _ in range(3)]
        
    def compute_barycenter_shift(self, true_img, predict_img):
        barycenter_true_X = [0, 0, 0]
        barycenter_true_Y = [0, 0, 0]
        barycenter_predict_X = [0, 0, 0]
        barycenter_predict_Y = [0, 0, 0]
        true_energy = [0, 0, 0]
        
        for i in range(56):
            for j in range(56):
                if true_img[i][j] > 1e-1:
                    energy_residual = (true_img[i][j] - predict_img[i][j]) / true_img[i][j]
                    if true_img[i][j] > 2 and true_img[i][j] <= 5:
                        self.energy_residual_2to5.append(energy_residual)
                        self.update_barycenter(i, j, true_img, predict_img, barycenter_true_X, barycenter_true_Y,
                                               barycenter_predict_X, barycenter_predict_Y, true_energy, 0)
                    if true_img[i][j] > 5 and true_img[i][j] <= 10:
                        self.energy_residual_5to10.append(energy_residual)
                        self.update_barycenter(i, j, true_img, predict_img, barycenter_true_X, barycenter_true_Y,
                                               barycenter_predict_X, barycenter_predict_Y, true_energy, 1)
                    elif true_img[i][j] > 10:
                        self.energy_residual_10toinfty.append(energy_residual)
                        self.update_barycenter(i, j, true_img, predict_img, barycenter_true_X, barycenter_true_Y,
                                               barycenter_predict_X, barycenter_predict_Y, true_energy, 2)
        self.calculate_barycenter_shift(barycenter_true_X, barycenter_true_Y, barycenter_predict_X, barycenter_predict_Y,
                                        true_energy)

    def update_barycenter(self, i, j, true_img, predict_img, barycenter_true_X, barycenter_true_Y,
                          barycenter_predict_X, barycenter_predict_Y, true_energy, idx):
        barycenter_true_X[idx] += i * true_img[i][j]
        barycenter_true_Y[idx] += j * true_img[i][j]
        barycenter_predict_X[idx] += i * predict_img[i][j]
        barycenter_predict_Y[idx] += j * predict_img[i][j]
        true_energy[idx] += true_img[i][j]

    def calculate_barycenter_shift(self, barycenter_true_X, barycenter_true_Y, barycenter_predict_X,
                                barycenter_predict_Y, true_energy):
        for k in range(3):
            # 避免除零错误，确保除数不为零
            if true_energy[k] != 0:
                barycenter_True_X = barycenter_true_X[k] / true_energy[k]
                barycenter_True_Y = barycenter_true_Y[k] / true_energy[k]
                barycenter_Predict_X = barycenter_predict_X[k] / true_energy[k]
                barycenter_Predict_Y = barycenter_predict_Y[k] / true_energy[k]

                self.barycenter_shift[k].append(np.sqrt((barycenter_True_X - barycenter_Predict_X) ** 2 +
                                                    (barycenter_True_Y - barycenter_Predict_Y) ** 2))

    def print_results(self):
        print(f"\tEnergy residual 2-5: {np.mean(self.energy_residual_2to5)}")
        print(f"\tEnergy residual 5-10: {np.mean(self.energy_residual_5to10)}")
        print(f"\tEnergy residual 10-infty: {np.mean(self.energy_residual_10toinfty)}")
        print(f"\tBarycenter shift 2-5: {np.mean(self.barycenter_shift[0])}")
        print(f"\tBarycenter shift 5-10: {np.mean(self.barycenter_shift[1])}")
        print(f"\tBarycenter shift 10-infty: {np.mean(self.barycenter_shift[2])}")


def process_model(model_name, model, test_dataloader, Y_max, Y_min, data_set):
    model.load_state_dict(torch.load(f'weights\\{model_name}.pth'))
    model.eval()
    
    weight_MSE_list = []
    predicted_images = []
    model_results = ModelResults()
    
    with torch.no_grad():
        for i, (X_test, Y_test) in enumerate(test_dataloader):
            outputs = model(X_test.to("cuda"))
            weight_MSE_list.append(weight_MSE(outputs.cpu().detach().numpy(), Y_test.numpy()) / len(Y_test))
            predicted_images.append(outputs.cpu().detach().numpy())
        
        print(f"Weighted MSE of {model_name}: {np.mean(weight_MSE_list)}")
    
    predicted_images = np.concatenate(predicted_images, axis=0)
    predicted_images = predicted_images * (Y_max - Y_min) + Y_min
    truth_list = data_set.val_Y.numpy()
    predict_list = predicted_images
    
    for true_img, predict_img in zip(truth_list, predict_list):
        true_img = true_img[0] * (Y_max - Y_min) + Y_min
        predict_img = predict_img[0]
        model_results.compute_barycenter_shift(true_img, predict_img)

    model_results.print_results()
    
    # Save results
    np.savetxt(f'plot\\{model_name}_energy_residual_2to5.txt', np.array(model_results.energy_residual_2to5))
    np.savetxt(f'plot\\{model_name}_energy_residual_5to10.txt', np.array(model_results.energy_residual_5to10))
    np.savetxt(f'plot\\{model_name}_energy_residual_10toinfty.txt', np.array(model_results.energy_residual_10toinfty))
    np.savetxt(f'plot\\{model_name}_barycenter_shift_2to5.txt', np.array(model_results.barycenter_shift[0]))
    np.savetxt(f'plot\\{model_name}_barycenter_shift_5to10.txt', np.array(model_results.barycenter_shift[1]))
    np.savetxt(f'plot\\{model_name}_barycenter_shift_10toinfty.txt', np.array(model_results.barycenter_shift[2]))


# 调用
for model_name, model in model_90k.items():
    process_model(model_name, model, test_dataloader, Y_max, Y_min, data_set)

'''''''''''''''''
计算jet数据集上的数字特征
'''''''''''''''''

# import os
# import numpy as np
# import pandas as pd

# # 设置文件夹路径
# folder_path = 'errors\\errors'

# # 初始化一个空的 DataFrame 来存储结果
# results = []

# # 遍历文件夹中的所有.txt文件
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)

#         # 读取文件内容
#         with open(file_path, 'r') as f:
#             numbers = [float(line.strip()) for line in f.readlines()]

#         # 计算特征
#         mean_value = np.mean(numbers)
#         variance_value = np.var(numbers)
#         std_value = np.std(numbers)
#         min_value = np.min(numbers)
#         max_value = np.max(numbers)

#         # 从文件名中提取信息
#         model_name, param_size, content = filename.replace('.txt', '').split('_')

#         # 将结果添加到列表中
#         results.append([model_name, param_size, content, mean_value, variance_value, std_value, min_value, max_value])

# # 将结果保存为 DataFrame
# df = pd.DataFrame(results, columns=['Model Name', 'Parameter Size', 'Content', 'Mean', 'Variance', 'Std Dev', 'Min', 'Max'])

# # 保存到 Excel 文件
# df.to_excel('output.xlsx', index=False)

# print('数据已保存到 output.xlsx')