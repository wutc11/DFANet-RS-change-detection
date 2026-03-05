# =============================================================================
# train_epoch = 50
# 一般
# =============================================================================
import argparse
import os
import pickle
import time

#import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from utils import *

from DDNet import DDNet

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('--data', metavar='DIR', default='.\\data\\Bern', help='path to image')
# parser.add_argument('--data', metavar='DIR', default='.\\data\\Sulzberger1', help='path to image')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['lenet','alexnet', 'vgg16', 'DDNet'], default='DDNet',
                    help='CNN architecture (default: lenet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')

parser.add_argument('--nmb_cluster', '--k', type=int, default=15,
                    help='number of cluster for k-means (default: 2)')



""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self,x_train, y_train):
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签

        # x=torch.FloatTensor(data_rotate(self.x_data[index].cpu().numpy()))
        # y=torch.FloatTensor(gasuss_noise(self.y_data[index]))
        # x=torch.FloatTensor(datarotate(self.x_data[index]))
        # return x,self.y_data[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestDS(torch.utils.data.Dataset):
    def __init__(self,x_train):
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签

        # x=torch.FloatTensor(data_rotate(self.x_data[index].cpu().numpy()))
        # y=torch.FloatTensor(gasuss_noise(self.y_data[index]))
        # x=torch.FloatTensor(datarotate(self.x_data[index]))
        # return x,self.y_data[index]
        return self.x_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def main():
    global args
    args = parser.parse_args()

## 训练数据处理
    import time
    import datetime
    start = datetime.datetime.now()
    from imageio import imread, imsave
    import matplotlib.pyplot as plt
    from scipy import io
    
    def extractPixelSamples(img, margin=1):
        """
        输入的img为单通道图像
        """
        X = img.copy()
        newX = np.zeros((
            X.shape[0] + 2 * margin,
            X.shape[1] + 2 * margin,
                  ))
        newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin] = X
        img = newX
        row, col = img.shape #得到图1的大小
        sampling_row = row - 2 * margin #忽略边缘未采样部分
        sampling_col = col - 2 * margin
        num_pixels = sampling_row * sampling_col #实际输入图像大小（299,299）
        sampleLen = (1 + 2*margin)**2 #窗口大小
        samples = np.zeros((num_pixels, sampleLen)) #（299*299,9）
        cnt = 0
        for i in range(margin, row - margin): #1-300
            for j in range(margin, col - margin):
                pixel_block = img[i-margin:i+margin+1, j-margin:j+margin+1]
                samples[cnt] = pixel_block.flatten() #逐像素3*3大小平铺成9，（299*299,9）
                cnt = cnt + 1
        return samples, sampling_row, sampling_col
    
    today = time.strftime("%Y-%m-%d+%H-%M-%S", time.localtime())
    print('Input SAR images and difference images.')

    # 数据输入
    # aug_train_labels_01
    # aug_train_samples_18
    # di_01
    # di_ori
    # gt_01
    # gt_ori
    # im1
    # im2
    # train_labels_01
    # train_samples_18
    data1 = "Yellow River I-SAR"
    data2 = "Yellow River II-SAR"
    data3 = "Yellow River III-SAR"
    data4 = "Yellow River IV-SAR"
    data5 = "Bern-SAR"
    data6 = "Ottawa-SAR"
    
    data7 = "Mexico City-optical"
    data8 = "Muragia-optical"
    
    data9 = "Guangzhou-vhr"
    data10 = "Shanghai-vhr"
    
    
    
    
#%% **数据修改
    train_epoch = 100
    file1, file2 = data3, data6
# =============================================================================
#     Yellow River I-SAR and Ottawa-SAR
# =============================================================================
    mat_file1 = io.loadmat('./images/'+file1+'.mat')
    mat_file2 = io.loadmat('./images/'+file2+'.mat')
    
    im1, im2, im3, im4, im5, im6 = mat_file1['im1'], mat_file1['im2'], mat_file1['gt_01'], \
        mat_file2['im1'], mat_file2['im2'], mat_file2['gt_01']
    
    t_samples, t_labels = mat_file1['aug_train_samples_18'], mat_file1['aug_train_labels_01'].reshape(-1)
    t_samples1, t_labels1 = mat_file2['aug_train_samples_18'], mat_file2['aug_train_labels_01'].reshape(-1)
    
    plt.imshow(im3, cmap='gray')
    plt.show()

    s = len(im1.shape)
    if s>2:
        if (im1[:, :, 1] == im1[:, :, 2]).all():
            im1 = im1[:, :, 1]    #防止读入（，，3）的多维数据
            print("im1yes")
        else:
            im1 = im1[:, :, 0]#+im1[:, :, 1]+im1[:, :, 2])/3
    s = len(im2.shape)
    if s>2:
        if (im2[:, :, 1] == im2[:, :, 2]).all():
            im2 = im2[:, :, 1]
            print("im2yes")
        else:
            im2 = im2[:, :, 0]#+im2[:, :, 1]+im2[:, :, 2])/3
    s = len(im3.shape)
    if s>2:
        if (im3[:, :, 1] == im3[:, :, 2]).all():
            im3 = im3[:, :, 1]    #防止读入（，，3）的多维数据
            print("im3yes")
   
    # 1. 提取筛选的im1和im2的数据
    col4 = t_samples[:, 0:9]   
    col13 = t_samples[:, 9:18]  
    
    # 2. 计算两列的差值
    difference = abs(col4/255 - col13/255)  # 得到形状为(m,9)的一维数组m,实际是di
    difference = difference.reshape(-1, 3, 3)  # 转换为(m, 3，3)的二维数组
    
    # 3. 复制三遍变成(m, 3, 3, 3)
    expanded = difference[..., np.newaxis]   # 形状变为 (m, 4, 4, 1)
    result = np.tile(expanded, (1, 1, 1, 3))   # 使用expand高效复制
    
    
    

    # x_train, y_train = createTrainingCubes(mdata, mlabel, patch_size)
    x_train = result.transpose(0, 3, 1, 2)
    y_train = t_labels
    print('... x train shape: ', x_train.shape) #(10000, 3, 7, 7)
    print('... y train shape: ', y_train.shape) #(74273, 3, 7, 7)


    # 创建 trainloader 和 testloader
    trainset = TrainDS(x_train, y_train)
    batch_size = 128
    # train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)



    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    istrain = True
    # 网络放到GPU上
    net = DDNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    net.train()
    
    # 开始训练
    from tqdm import tqdm
    
    if os.path.exists('./results/5TSPLR/' + file1 +'_' + args.arch +'_model.pth'):
        ## 读取模型
        model_load = net
        print("already trained")
        state_dict = torch.load('./results/5TSPLR/' + file1 +'_' + args.arch +'_model.pth')
        model_load.load_state_dict(state_dict['model'])
    else:
        # 开始训练
        total_loss = 0
        for epoch in tqdm(range(train_epoch)):
            for i, (inputs, labels) in enumerate(train_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)
                # 优化器梯度归零
                optimizer.zero_grad()
                # 正向传播 +　反向传播 + 优化
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('[Epoch: %d]  [loss avg: %.4f]  [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
        print('Finished Training')

        model = net
        torch.save({'model': model.state_dict()}, './results/5TSPLR/' + file1 +'_' + args.arch +'_model.pth')




# %% 测试
    samples_1, srow, scol = extractPixelSamples(im1)
    # expanded_1 = np.concatenate((samples_1, samples_1[:(samples_1.shape[0]//100*100+100-samples_1.shape[0])]), axis=0)
    samples_2, _, _ = extractPixelSamples(im2)
    # expanded_2 = np.concatenate((samples_2, samples_2[:(samples_2.shape[0]//100*100+100-samples_2.shape[0])]), axis=0)
    difference = abs(samples_2/255 - samples_1/255)  # 得到形状为(m,9)的一维数组m,实际是di
    difference = difference.reshape(-1, 3, 3)  # 转换为(m, 3，3)的二维数组
    
    # 3. 复制三遍变成(m, 3, 3, 3)
    expanded = difference[..., np.newaxis]   # 形状变为 (m, 4, 4, 1)
    all_data = np.tile(expanded, (1, 1, 1, 3))   # 使用expand高效复制
    x_test = all_data.transpose(0, 3, 1, 2)
    print('... x test shape: ', x_test.shape)
    
    
    testset = TestDS(x_test)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=x_test.shape[0], shuffle=False, num_workers=0)
    
    net.eval()
    count = 0
    for inputs_1 in test_loader:

        inputs_1 = inputs_1.to(device)
        with torch.no_grad(): # 在测试的时候必须加上这行和下一行代码，否则预测会出问题，这里是防止还有梯度更新这些，而且如果不加这个，后面又没有进行梯度更新的话，可能会报显存不够用的错误，我怀疑是数据没有被清理
            net.eval() # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
            outputs = net(inputs_1)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)              
                
                
    end = datetime.datetime.now()
    CDMap = np.reshape(outputs, [im3.shape[0], im3.shape[1]])
    CDMap = np.uint8(CDMap*255)
    
    plt.imshow(CDMap, cmap='gray')
    plt.show()



#%% 结果分析
##  比较系数  ########################"""
    print("\npcc and kappa are ok,congraulation!!!")
    print("CDMap:",CDMap.shape)
    ref = im3 * 255 # zz4为真相图的0/1值图
    ref = np.array(ref,'uint8')
    image = CDMap
    
    import sklearn.metrics as mt     
    mask = ref/255
    CD_map = image/255
    
    [[zTN, zFP], [zFN, zTP]] = mt.confusion_matrix(mask.flatten(), CD_map.flatten()) #计算混淆矩阵来评估分类的准确性。
    zOA = mt.accuracy_score(mask.flatten(), CD_map.flatten()) #准确性分类分数。
    zKappa = mt.cohen_kappa_score(mask.flatten(), CD_map.flatten()) #计算Cohen’s kappa：一个衡量注释者间一致性的统计数据。
    
    zPREC_0 = mt.precision_score(mask.flatten(), CD_map.flatten(), pos_label=0) #查准率或者精度； precision(查准率)=TP/(TP+FP)
    zPREC_1 = mt.precision_score(mask.flatten(), CD_map.flatten())
    zREC_0 = mt.recall_score(mask.flatten(), CD_map.flatten(), pos_label=0) #查全率 ；recall(查全率)=TP/(TP+FN)
    zREC_1 = mt.recall_score(mask.flatten(), CD_map.flatten())
    
    zAUC = mt.roc_auc_score(mask.flatten(), CD_map.flatten()) #从预测分数计算接收者工作特征曲线下的面积（ROC AUC）。
    zF1_Score = mt.f1_score(mask.flatten(), CD_map.flatten()) #计算F1分数，也称为平衡f分数或f测量。
    zOA, zKappa, zAUC, zF1_Score \
        = round(zOA, 4), round(zKappa, 4), round(zAUC, 4), round(zF1_Score, 4)
    
    imsave('./results/5TSPLR/' + today + file1 + '_0255.bmp', CDMap)
    # imsave('./results/proposed/' + today + file2 + '_0255.bmp', CDMap1)
    
    f = open('./results/5TSPLR/' + file1 + '.txt', 'a')
    f.write('\n'+'Start Time: '+ str(start))
    f.write('\n'+'batch_size: '+ str(batch_size))
    f.write('\n' + today + file1 + ' TN: '+ str(zTN) + ' TP: '+ str(zTP) + ' FN: '+ str(zFN)+' FP: '+ str(zFP)+' OA: '+ str(zOA)+' Kappa: '+ str(zKappa)+' AUC: '+ str(zAUC)+' F1: '+ str(zF1_Score))
    # f.write('\n' + today + file2 +' FN1:'+ str(zzz3FN1)+' FP1: '+ str(zzz4FP1)+' OE1: '+ str(zzz5OE1)+' PCC1: '+ str(zzz6PCC1)+' KC1: '+ str(zzz8KAPPA1))
    f.write('\n')
    f.write('Time: {} '.format((end - start)))  # 时间
    f.write('\n')
    f.close()  
  
    
    
if __name__ == '__main__':
    main()


'''
yellow_river
Change detection results ==>
 ... ... FP:   1291
 ... ... FN:   2333
 ... ... OE:   3624
 ... ... PCC:  95.12
 ... ... KC:  95.42
'''

'''
ottawa
85451
16049
 Change detection results ==>
 ... ... FP:   762
 ... ... FN:   885
 ... ... OE:   1647
 ... ... PCC:  98.38
 ... ... KC:  98.52
'''

'''
52926
12610
 Change detection results ==>
 ... ... FP:   265
 ... ... FN:   654
 ... ... OE:   919
 ... ... PCC:  98.60
 ... ... KC:  98.93
'''