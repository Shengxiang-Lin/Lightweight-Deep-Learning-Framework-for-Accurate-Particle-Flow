import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import random
from math import pi
import os
import sys

name1 = 'CNN3Dver2_90k'
name2 = 'pts'
output_txt = 'errors/' + name1 + '_' + name2 + '.txt'
input_predicted_txt = 'result/predicted/result_' + name1 + '_' + 'predicted.txt'

sys.stdout=open(output_txt,'w')
def chord_length(x1, y1, x2, y2, r):
    # 计算两点间的弦长
    L = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 计算圆心到弦的垂直距离
    d = math.sqrt(r ** 2 - (L / 2) ** 2)

    # 使用余弦定理计算弦所对的圆心角
    cos_theta = (L ** 2 + r ** 2 - d ** 2) / (2 * L * r)
    theta = math.acos(cos_theta)

    # 使用圆心角和半径计算圆弧长度
    s = theta * r

    return s
def calculate_spatial_coordinates(rap, phi, Pt, E):
    # Calculate the momentum components
    p_T = Pt  # This is the transverse momentum
    p_x = p_T * math.cos(phi)
    p_y = p_T * math.sin(phi)
    p_z = E * math.sinh(rap)  # Assuming the speed of light is 1 (c = 1) and the time is 0 (t = 0)

    # Calculate the spatial coordinates
    x = p_x / E  # c * t is assumed to be 1
    y = p_y / E  # c * t is assumed to be 1
    z = p_z / E  # c * t is assumed to be 1

    return x, y, z

def plot_filled_circle_in_3d(px=0, py=0, area=0, height=0, color="black"):
    theta = np.linspace(0, 2 * np.pi, 100)  # 角度分割
    radius= math.sqrt(area/2/pi)
    # 设置坐标轴范围
    ax.set_xlim(-3, 3)  # 你可以根据需要调整这些值
    ax.set_ylim(-1, 7)
    ax.set_zlim(0, 200)  # 假设height是已知的，可以根据需要调整z轴的上限
    x = radius * np.cos(theta)+px  # x坐标
    y = radius * np.sin(theta)+py  # y坐标
    z = np.zeros_like(x)  # z坐标，圆在xoy平面上
    # 绘制圆的边界
   # color = (random.random(), random.random(), random.random())  # 随机颜色
    ax.plot(x, y, z, color=color)

    # 为了填充圆内的区域，我们需要对每个角度进行操作
    for t in theta:
        # 计算圆内点的坐标
        x_fill = np.linspace(-radius, radius, 50)
        y_fill = np.sqrt(radius ** 2 - x_fill ** 2)  # y坐标的上半部分
        z_fill = np.zeros_like(x_fill)

        # 绘制圆的上半部分
        ax.plot_surface(np.array([x_fill+px, x_fill+px]), np.array([y_fill+py, -y_fill+py]), np.array([z_fill, z_fill]),
                        color=color, alpha=0.5)
    #绘制圆柱
    x = 0.1*np.cos(theta)+px  # x坐标
    y = 0.1*np.sin(theta)+py  # y坐标
    z = np.linspace(0, height, 100)  # z坐标
    X, Z = np.meshgrid(x, z)
    Y = np.meshgrid(y, z)[0]
    ax.plot_surface(X, Y, Z, color=color, alpha=0.8)

    # 创建圆柱的顶面和底面
    Z_top = np.full_like(X, z[-1])
    Z_bottom = np.full_like(X, z[0])
    ax.plot_surface(X, Y, Z_top, color=color, alpha=0.5)
    ax.plot_surface(X, Y, Z_bottom, color=color, alpha=0.5)

    # 设置坐标轴标签
    ax.set_xlabel('y')
    ax.set_ylabel('φ')
    ax.set_zlabel('Pt(GeV)')

# 打开文件
with open('result/result_truth.txt', 'r') as file:
    # 读取文件内容
    content = file.read()
    # 分割内容为行
    lines = content.strip().split('\n')
    # 读取每行的数字
    truth_numbers = []
    for line in lines:
        # 分割行中的数字
        truth_numbers_in_line = line.split()

        # 将数字转换为浮点数列表
        truth_numbers.extend(map(float, truth_numbers_in_line))

    # 打印所有数字
    #for number in numbers:
        #print(number)
truth_numbers_size = len(truth_numbers)
#print(truth_numbers_size)
num=0
truth_clusters= []
truth_raps= []
truth_phis= []
truth_pts= []
truth_E= []
truth_areas= []
truth_x= []
truth_y= []
truth_z= []
p=0
for i in range(truth_numbers_size):
    if p!=0:
        p=p-1
        continue
    if num==0:
        truth_clusters.append(int(truth_numbers[i]))
        num=int(truth_numbers[i])
        continue
    num=num-1;
    truth_raps.append(truth_numbers[i])
    truth_phis.append(truth_numbers[i+1])
    truth_pts.append(truth_numbers[i+2])
    truth_E.append(truth_numbers[i+3])
    truth_areas.append(truth_numbers[i+4])
    y, x, z = calculate_spatial_coordinates(truth_numbers[i], truth_numbers[i+1], truth_numbers[i+2],truth_numbers[i+3])
    truth_y.append(y)
    truth_x.append(x)
    truth_z.append(z)
    p=4
truth_clusters_size = len(truth_clusters)
#print(truth_clusters_size)
truth_pos=0
truth_clusters[-1]=0


with open('result/result_jet.txt', 'r') as file:
    # 读取文件内容
    content = file.read()
    # 分割内容为行
    lines = content.strip().split('\n')
    # 读取每行的数字
    jet_numbers = []
    for line in lines:
        # 分割行中的数字
        jet_numbers_in_line = line.split()

        # 将数字转换为浮点数列表
        jet_numbers.extend(map(float, jet_numbers_in_line))

    # 打印所有数字
    #for number in numbers:
        #print(number)
jet_numbers_size = len(jet_numbers)
#print(jet_numbers_size)
num=0
jet_clusters= []
jet_raps= []
jet_phis= []
jet_pts= []
jet_E= []
jet_areas= []
jet_x= []
jet_y= []
jet_z= []
p=0
for i in range(jet_numbers_size):
    if p!=0:
        p=p-1
        continue
    if num==0:
        jet_clusters.append(int(jet_numbers[i]))
        num=int(jet_numbers[i])
        continue
    num=num-1;
    jet_raps.append(jet_numbers[i])
    jet_phis.append(jet_numbers[i+1])
    jet_pts.append(jet_numbers[i+2])
    jet_E.append(jet_numbers[i+3])
    jet_areas.append(jet_numbers[i+4])
    y, x, z = calculate_spatial_coordinates(jet_numbers[i], jet_numbers[i+1], jet_numbers[i+2],jet_numbers[i+3])
    jet_y.append(y)
    jet_x.append(x)
    jet_z.append(z)
    p=4
jet_clusters_size = len(jet_clusters)
#print(jet_clusters_size)
jet_pos=0
jet_clusters[-1]=0

with open(input_predicted_txt, 'r') as file:
#with open('old/result_predict.txt', 'r') as file:
    # 读取文件内容
    content = file.read()
    # 分割内容为行
    lines = content.strip().split('\n')
    # 读取每行的数字
    predict_numbers = []
    for line in lines:
        # 分割行中的数字
        predict_numbers_in_line = line.split()

        # 将数字转换为浮点数列表
        predict_numbers.extend(map(float, predict_numbers_in_line))

    # 打印所有数字
    #for number in numbers:
        #print(number)
predict_numbers_size = len(predict_numbers)
#print(f'predict_numbers_size:{predict_numbers_size}')
num=0
predict_clusters= []
predict_raps= []
predict_phis= []
predict_pts= []
predict_E= []
predict_areas= []
predict_x= []
predict_y= []
predict_z= []
p=0
for i in range(predict_numbers_size):
    if p!=0:
        p=p-1
        continue
    if num==0:
        predict_clusters.append(int(predict_numbers[i]))
        num=int(predict_numbers[i])
        continue
    num=num-1;
    predict_raps.append(predict_numbers[i])
    predict_phis.append(predict_numbers[i+1])
    predict_pts.append(predict_numbers[i+2])
    predict_E.append(predict_numbers[i+3])
    predict_areas.append(predict_numbers[i+4])
    y, x, z = calculate_spatial_coordinates(predict_numbers[i], predict_numbers[i+1], predict_numbers[i+2],predict_numbers[i+3])
    predict_y.append(y)
    predict_x.append(x)
    predict_z.append(z)
    p=4
predict_clusters_size = len(predict_clusters)
#print(predict_clusters_size)
predict_pos=0
predict_clusters[-1]=0

def distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points in 2D space."""
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
def cos_between_vectors(x1,y1,x2, y2):
    # 计算点积
    dot_product = x1 * x2 + y1 * y2
    # 计算两个向量的模
    magnitude_v1 = math.sqrt(x1**2 + y1**2)
    magnitude_v2 = math.sqrt(x2**2 + y2**2)
    # 计算夹角的余弦值
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return cos_angle

output_folder='output_figure'
for i in range(truth_clusters_size):
    truth_pos=truth_pos+truth_clusters[i-1]
    #fig = plt.figure()
    #print(truth_clusters[i])
    #ax = fig.add_subplot(111, projection='3d')
    jet_pos = jet_pos + jet_clusters[i - 1]
    predict_pos = predict_pos + predict_clusters[i - 1]
    if jet_clusters[i]==0:
        continue
    #for j in range(jet_clusters[i]):
       #plot_filled_circle_in_3d(px=jet_raps[j+jet_pos], py=jet_phis[j+jet_pos], area=jet_areas[j+jet_pos], height=jet_pts[j+jet_pos])
       #print(f'{jet_raps[j + jet_pos]}  {jet_phis[j + jet_pos]}  {jet_pts[j + jet_pos]}  {jet_areas[j + jet_pos]}')
    truth_min_cos =-1
    predict_min_cos = -1
    truth_min_distance = math.sqrt(56*56+56*56)
    predict_min_distance = math.sqrt(56 * 56 + 56 * 56)
    truth_id=0
    for j in range(truth_clusters[i]):
       #plot_filled_circle_in_3d(px=truth_raps[j+ truth_pos], py=truth_phis[j+ truth_pos], area=truth_areas[j+ truth_pos],height=truth_pts[j+ truth_pos], color="red")
       #if truth_min_cos<cos_between_vectors(x1=truth_x[j+ truth_pos],y1=truth_y[j+ truth_pos],x2=jet_x[jet_pos],y2=jet_y[jet_pos]):
          # truth_id=j
          # truth_min_cos = cos_between_vectors(x1=truth_x[j + truth_pos], y1=truth_y[j + truth_pos],x2=jet_x[jet_pos], y2=jet_y[jet_pos])
       if truth_min_distance>distance(x1=truth_raps[j+ truth_pos],y1=truth_phis[j+ truth_pos],x2=jet_raps[jet_pos],y2=jet_phis[jet_pos]):
           truth_id=j
           truth_min_distance = distance(x1=truth_raps[j + truth_pos], y1=truth_phis[j + truth_pos], x2=jet_raps[jet_pos],y2=jet_phis[jet_pos])
    for j in range(predict_clusters[i]):
       #plot_filled_circle_in_3d(px=predict_raps[j+ predict_pos], py=predict_phis[j+ predict_pos], area=truth_areas[j+ predict_pos], height=predict_pts[j+ predict_pos], color="green")
       #if predict_min_cos < cos_between_vectors(x1=predict_x[j + predict_pos], y1=predict_y[j + predict_pos],x2=jet_x[jet_pos], y2=jet_y[jet_pos]):
         #  predict_id = j
         #  predict_min_cos = cos_between_vectors(x1=predict_x[j + predict_pos], y1=predict_y[j + predict_pos],x2=jet_x[jet_pos], y2=jet_y[jet_pos])
       if predict_min_distance > distance(x1=predict_raps[j + predict_pos], y1=predict_phis[j + predict_pos],x2=jet_raps[jet_pos], y2=jet_phis[jet_pos]):
           predict_id = j
           predict_min_distance = distance(x1=predict_raps[j + predict_pos], y1=predict_phis[j + predict_pos],x2=jet_raps[jet_pos], y2=jet_phis[jet_pos])
    #print(f'{jet_raps[jet_pos]}  {jet_raps[jet_pos]}  {jet_pts[jet_pos]}  {jet_areas[jet_pos]}')
    #print(f'{truth_raps[truth_id+truth_pos]}  {truth_phis[truth_id+truth_pos]}  {truth_pts[truth_id+truth_pos]}  {truth_areas[truth_id+truth_pos]}')
    #print(f'{predict_raps[predict_id + predict_pos]}  {predict_phis[predict_id + predict_pos]}  {predict_pts[predict_id + predict_pos]}  {predict_areas[predict_id + predict_pos]}')
    #print(f'{truth_min_cos}  {predict_min_cos}')
    #y, x, z = calculate_spatial_coordinates(jet_raps[jet_pos], jet_raps[jet_pos], jet_pts[jet_pos], jet_E[jet_pos])
    x=jet_y[jet_pos]
    y=jet_x[jet_pos]
    z=jet_z[jet_pos]
#    print(f'x:{x*r/math.sqrt(x*x+y*y)} y:{y*r/math.sqrt(x*x+y*y)}')
    r=56/2/math.pi
    y=-y;
    #print(f'x:{x} y:{y} z:{z}')
    #print(f'x:{x * r / math.sqrt(x * x + y * y)} y:{y * r / math.sqrt(x * x + y * y)}')
    #dis = chord_length(0, r, x*r/math.sqrt(x*x+y*y), y*r/math.sqrt(x*x+y*y), r)
    theta=math.atan(x/math.fabs(y))
    dis=(math.pi-theta)*r
    #print(f'x:{x*r/math.sqrt(x*x+y*y)}  y:{y*r/math.sqrt(x*x+y*y)} width:{z+28}  r:{x*x+y*y}');
    #print(f'width:{dis}  height:{28+z}');

    #y, x, z = calculate_spatial_coordinates(truth_raps[jet_pos], truth_raps[jet_pos], truth_pts[jet_pos], truth_E[jet_pos])
    x=truth_y[truth_id+truth_pos]
    y=truth_x[truth_id+truth_pos]
    z=truth_z[truth_id+truth_pos]
    r = 56 / 2 / math.pi
    y = -y;
    #print(f'x:{x} y:{y} z:{z}')
    # dis = chord_length(0, r, x*r/math.sqrt(x*x+y*y), y*r/math.sqrt(x*x+y*y), r)
    theta = math.atan(x / math.fabs(y))
    dis = (math.pi - theta) * r
    # print(f'x:{x*r/math.sqrt(x*x+y*y)}  y:{y*r/math.sqrt(x*x+y*y)} width:{z+28}  r:{x*x+y*y}');
    #print(f'width:{dis}  height:{28 + z}');
    dis1=dis
    z1=z
    x1=x
    y1=y
  #  y, x, z = calculate_spatial_coordinates(predict_raps[jet_pos], predict_raps[jet_pos], predict_pts[jet_pos], predict_E[jet_pos])
    x=predict_y[predict_id+predict_pos]
    y=predict_x[predict_id+predict_pos]
    z=predict_z[predict_id+predict_pos]
    r = 56 / 2 / math.pi
    y = -y;
    #print(f'x:{x} y:{y} z:{z}')
    # dis = chord_length(0, r, x*r/math.sqrt(x*x+y*y), y*r/math.sqrt(x*x+y*y), r)
    theta = math.atan(x / math.fabs(y))
    dis = (math.pi - theta) * r
    # print(f'x:{x*r/math.sqrt(x*x+y*y)}  y:{y*r/math.sqrt(x*x+y*y)} width:{z+28}  r:{x*x+y*y}');
    #print(f'width:{dis}  height:{28 + z}');
    #print(f'{math.sqrt((dis1-dis)*(dis1-dis)+(z1-z)*(z1-z))}')
    if name2 == 'dis':
        print(f'{math.sqrt((x1 - x) * (x1 -x) + (y1-y)*(y1-y)+(z1 - z) * (z1 - z))}')
    if name2 == 'pts':
        print(f'{(predict_E[predict_id+predict_pos]-truth_E[truth_id+truth_pos])/truth_E[truth_id+truth_pos]}')
    #print(f'{truth_pts[truth_id+truth_pos]}   {predict_pts[predict_id+predict_pos]}')
    #plt.savefig(os.path.join(output_folder, f'{int(i)}_predict.png'))
   # plt.close()  # 关闭当前图形

