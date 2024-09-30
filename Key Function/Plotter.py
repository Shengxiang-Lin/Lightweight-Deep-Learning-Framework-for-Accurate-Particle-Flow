#绘制直方图模块
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
class Plotter:
    def __init__(self):
        pass
    def plot_histogram(self, data_list, num_bins=10, color='blue', title='Histogram',xlabel='Values',ylabel='Frequency',net = 'None',
                    text = 'None',need_log_scale=False,range=[-1,-1],image_info = 'None',data_points='0'):
        # 计算范围
        data_min, data_max = np.min(data_list), np.max(data_list)
        if range == [-1,-1]:
            data_range = (data_min, data_max)
        else :
            data_range = range
        hist, bin_edges=np.histogram(data_list,bins=num_bins,range=data_range)
        max_bin_index = np.argmax(hist)
        max_bin_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])
        max_bin_freq = hist[max_bin_index] # 该 bin 的频率
        print(f"频率最高的 bin 的范围是: {max_bin_range}，频率为: {max_bin_freq}")
        plt.hist(data_list,bins=num_bins,range=data_range,histtype='stepfilled',align='mid',orientation='vertical',color=color)
        if need_log_scale:
            plt.yscale('log')
        plt.title(title,fontsize=17)
        plt.xlabel(xlabel,fontsize=12)
        plt.tick_params(labelsize=12)
        plt.ylabel(ylabel,fontsize=12)
        #在图中添加文本
        plt.text(0.5, -0.2, f"The most frequent data bin: {max_bin_range}\nThe std of the data is {np.std(data_list)}",
                ha='center', va='center', transform=plt.gca().transAxes,fontsize=12)
        if text != 'None':
            plt.text(0.5, -0.3, text, ha='center', va='center', transform=plt.gca().transAxes,fontsize=12)
        if image_info != 'None':
            plt.text(0.5, -0.4, 'Generated from '+image_info+ '. The num of data points is '+ data_points, ha='center', va='center', 
                    transform=plt.gca().transAxes,fontsize=12)
        if net != 'None':
            plt.text(0.5, -0.5, 'The network is '+net, ha='center', va='center', 
                    transform=plt.gca().transAxes,fontsize=12)
    
    def plot_scatter(self,list1,list2,title='scatter',color='r',s=1,xlabel='xlabel',ylabel='ylabel', text = 'None',net = 'None',
                    image_info = 'None',data_points='0'):
        fig, ax1=plt.subplots(figsize=(10,10))
        # 绘制对角线
        ax1.plot([min(list1), max(list1)], [min(list1), max(list1)], 'k--', label='X=Y')
        # 绘制散点图
        ax1.scatter(list1, list2, s=s, color=color)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        plt.tick_params(labelsize=12)
        ax1.set_title(title)
        # 添加文本
        plt.text(0.5, -0.1, text, ha='center', va='center', transform=plt.gca().transAxes)
        # 拟合散点图
        coefficients = np.polyfit(list1, list2, 1)
        polynomial = np.poly1d(coefficients)
        x_fit = np.linspace(min(list1), max(list1), 100)
        y_fit = polynomial(x_fit)
        if text != 'None':
            plt.text(0.5, -0.3, text, ha='center', va='center', transform=plt.gca().transAxes,fontsize=12)
        if image_info != 'None':
            plt.text(0.5, -0.2, 'Generated from '+image_info+ '. The num of data points is '+ data_points, ha='center', va='center', 
                    transform=plt.gca().transAxes,fontsize=12)
        if net != 'None':
            plt.text(0.5, -0.4, 'The network is '+net, ha='center', va='center', transform=plt.gca().transAxes,fontsize=12)
        # 绘制最小二乘拟合线
        ax1.plot(x_fit, y_fit, 'r-', label='Least Squares Fit', linewidth=2, alpha=0.7,color='b')
        
    def plot_error_pixel_map(self, true_list, predict_list, HIGTHT=56, WIDTH=56,text = 'None',net='None',
                    image_info = 'None',data_points='0'):
        # 计算误差矩阵
        Error_pixel_map = np.array(true_list) - np.array(predict_list)
        Error_pixel_map = Error_pixel_map.astype(np.float32).reshape(HIGTHT, WIDTH)
        # 创建颜色映射
        plt.tick_params(labelsize=12)
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['red', 'white', 'black'])
        norm = mcolors.TwoSlopeNorm(vmin=Error_pixel_map.min(), vcenter=0, vmax=Error_pixel_map.max())

        # 绘制误差像素图
        plt.figure(figsize=(10, 5))
        plt.imshow(Error_pixel_map, cmap=cmap, norm=norm)
        plt.colorbar(label='Error Value')
        plt.title('Error Pixel Map')
        if text != 'None':
            plt.text(0.5, -0.1, text, ha='center', va='center', transform=plt.gca().transAxes,fontsize=12)
        if image_info != 'None':
            plt.text(0.5, -0.2, 'Generated from '+image_info+ '. The num of data points is '+ data_points, ha='center', va='center', 
                    transform=plt.gca().transAxes,fontsize=12)
        if net != 'None':
            plt.text(0.5, -0.3, 'The network is '+net, ha='center', va='center', 
                    transform=plt.gca().transAxes,fontsize=12)
        plt.tight_layout()
        plt.show()