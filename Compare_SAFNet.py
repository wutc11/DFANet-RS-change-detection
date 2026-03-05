# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:30:13 2025
train_epoch 大概40代就收敛了
@author: PC
"""

#%% 网络定义
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import os
import random
from skimage import io, measure

# Setting the seed of GPU

# def seed_torch(seed = 123):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed) 
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# seed_torch()

# 1. Feature extraction network
class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(16)
        
        self.conv1_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 1 ,2, 1)

        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 1, 2, 1)

        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(64)

        # Feature fusion
        self.conv_fusion1 = nn.Conv2d(16, 64, 1, 4, 2)
        self.conv_fusion2 = nn.Conv2d(32, 64, 1, 2, 1)

    def forward(self, x):

        x1 = self.conv1(x)
        x = F.relu(self.bn1_1(self.conv1_1(x1)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x)))
        x = x1 + x_1
        x2 = self.conv2(x)
        x = F.relu(self.bn2_1(self.conv2_1(x2)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x)))
        x = x2 + x_2
        x3 = self.conv3(x)
        x = F.relu(self.bn3_1(self.conv3_1(x3)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x)))
        return x_1, x_2, x_3

class FeatFuse(nn.Module):
    def __init__(self):
        super(FeatFuse, self).__init__()

        self.conv_fusion1 = nn.Conv2d(16, 64, 1, 4, 3)
        self.conv_fusion2 = nn.Conv2d(32, 64, 1, 2, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 64*3)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        
        batch_size = x1.size(0)
        out_channels = x3.size(1)
        x1 = self.conv_fusion1(x1)
        x2 = self.conv_fusion2(x2)
        output = []
        output.append(x1)
        output.append(x2)
        output.append(x3)
        x = x1 + x2 + x3
          
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        a_b = x.reshape(batch_size, 3, out_channels, -1)
        a_b = self.softmax(a_b)
        #the part of selection
        a_b = list(a_b.chunk(3, dim=1))#split to a and b
        a_b = list(map(lambda x:x.reshape(batch_size, out_channels, 1, 1), a_b))
        V = list(map(lambda x,y:x*y, output, a_b))
        V = reduce(lambda x,y:x+y, V)
        return V
    
    
# 2. The proposed SAFNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.featnet = FeatNet()
        self.featfuse = FeatFuse()
        self.featnet1 = FeatNet()
        self.featfuse1 = FeatFuse()
        #self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)
        
        self.global_pool1 = nn.AdaptiveAvgPool2d(1)
        self.global_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(64, 2)
      

    def forward(self, x, y):

        x1_1, x1_2, x1_3 = self.featnet(x)
        x2_1, x2_2, x2_3 = self.featnet1(y)

        feat_11 = self.featfuse(x1_1, x1_2, x1_3)
        feat_22 = self.featfuse1(x2_1, x2_2, x2_3)
        feat_1 = self.global_pool1(feat_11)
        feat_2 = self.global_pool2(feat_22)
        feat_1 = feat_1.view(feat_1.size(0), -1)
        feat_2 = feat_2.view(feat_2.size(0), -1)
        feat_1 = self.fc1(feat_1)
        feat_2 = self.fc2(feat_2)

        feature_corr = self.xcorr_depthwise(feat_11, feat_22)
        feat = feature_corr.view(feature_corr.size(0), -1)
        #feat = global_pool(feature_corr)
        feat = self.fc(feat)
        return feat_1, feat_2, feat

    def xcorr_depthwise(self, x, kernel):

        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    
# 3. Data processing function

def addZeroPadding(X, margin=2):
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]
              ))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX


def createImgCube(X ,gt ,pos:list ,windowSize=25):
    margin = (windowSize-1)//2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    dataPatches = np.zeros((pos.__len__(), windowSize, windowSize, X.shape[2]))
    if( pos[-1][1]+1 != X.shape[1] ):
        nextPos = (pos[-1][0] ,pos[-1][1]+1)
    elif( pos[-1][0]+1 != X.shape[0] ):
        nextPos = (pos[-1][0]+1 ,0)
    else:
        nextPos = (0,0)
    return np.array([zeroPaddingX[i:i+windowSize, j:j+windowSize, :] for i,j in pos ]),\
    np.array([gt[i,j] for i,j in pos]) ,\
    nextPos


def createPos(shape:tuple, pos:tuple, num:int):
    if (pos[0]+1)*(pos[1]+1)+num >shape[0]*shape[1]:
        num = shape[0]*shape[1]-( (pos[0])*shape[1] + pos[1] )
    return [(pos[0]+(pos[1]+i)//shape[1] , (pos[1]+i)%shape[1] ) for i in range(num) ]


def createPosWithoutZero(hsi, gt):

    mask = gt > 0
    return [(i,j) for i , row  in enumerate(mask) for j , row_element in enumerate(row) if row_element]


def splitTrainTestSet(X, gt, testRatio, randomState=111):

    X_train, X_test, gt_train, gt_test = train_test_split(X, gt, test_size=testRatio, random_state=randomState, stratify=gt)
    return X_train, X_test, gt_train, gt_test


def createImgPatch(lidar, pos:list, windowSize=25):

    margin = (windowSize-1)//2
    zeroPaddingLidar = np.zeros((
      lidar.shape[0] + 2 * margin,
      lidar.shape[1] + 2 * margin
            ))
    zeroPaddingLidar[margin:lidar.shape[0]+margin, margin:lidar.shape[1]+margin] = lidar
    return np.array([zeroPaddingLidar[i:i+windowSize, j:j+windowSize] for i,j in pos ])


def minmax_normalize(array):    
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num+1):
        idy, idx = np.where(res==i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new




# %% **数据集定义
if __name__=="__main__":
## 训练数据处理
    # 4. Create dataloader
    windowSize = 3 # patch size
    class_num = 2
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
    
    file1, file2 = data3, data6
    train_epoch = 5
    # =============================================================================
    #     Yellow River I-SAR and Ottawa-SAR
    # =============================================================================
    mat_file1 = io.loadmat('./images/'+file1+'.mat')
    mat_file2 = io.loadmat('./images/'+file2+'.mat')
    
    im1, im2, im3, im4, im5, im6 = mat_file1['im1'], mat_file1['im2'], mat_file1['gt_01'], \
        mat_file2['im1'], mat_file2['im2'], mat_file2['gt_01']
    
    t_samples, t_labels = mat_file1['aug_train_samples_18'], mat_file1['aug_train_labels_01'].reshape(-1)
    t_samples1, t_labels1 = mat_file2['aug_train_samples_18'], mat_file2['aug_train_labels_01'].reshape(-1)
    
    di, di1 = mat_file1['di_01'], mat_file2['di_01']
    plt.imshow(im3, cmap='gray')
    plt.show()
    plt.imshow(di, cmap='gray')
    plt.show()

    # 预处理图像（处理多通道情况）
    def preprocess_image(img):
        if len(img.shape) > 2:
            if img.shape[2] == 3:
                # 如果三个通道相同，取第一个通道
                if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2]):
                    return img[:, :, 0]
                else:
                    # 否则转换为灰度
                    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
            else:
                return img[:, :, 0]  # 只取第一个通道
        return img
    
    im1 = preprocess_image(im1)
    im2 = preprocess_image(im2)
    im4 = preprocess_image(im4)
    im5 = preprocess_image(im5)    
       
    # 1. 提取筛选的im1和im2的数据
    col4 = t_samples[:, 0:9]   
    col13 = t_samples[:, 9:18]  
    
    # 2. 分别得到im1,2
    get_im1 = col4.reshape(-1, 3, 3)   # 得到形状为(m,9)的一维数组m,实际是di
    get_im2 = col13.reshape(-1, 3, 3)  # 转换为(m, 3，3)的二维数组
    
    # 3. 拓展变成(m, 3, 3, 1)
    train_1 = get_im1[..., np.newaxis]   # 形状变为 (m, 4, 4, 1)
    train_2 = get_im2[..., np.newaxis] 
    
    
    X_train = torch.from_numpy(train_1.transpose(0,3,1,2)).float()
    X_train_2 = torch.from_numpy(train_2.transpose(0,3,1,2)).float()
    train_labels = t_labels+1
    
    
    # 准备所有test数据
    samples_1, srow, scol = extractPixelSamples(im1)
    samples_2, _, _ = extractPixelSamples(im2)
    X_test, X_test_2 = samples_1.reshape(-1, 3, 3), samples_2.reshape(-1, 3, 3)
    X_test, X_test_2 = X_test[..., np.newaxis], X_test_2[..., np.newaxis]
    
    X_test = torch.from_numpy(X_test.transpose(0,3,1,2)).float()
    X_test_2 = torch.from_numpy(X_test_2.transpose(0,3,1,2)).float()
    
    print (X_train.shape)
    print (X_test.shape)
    print("Creating dataloader")
    
    """ Training dataset"""
    class TrainDS(torch.utils.data.Dataset):
        def __init__(self):
            self.len = train_labels.shape[0]
            self.hsi = torch.FloatTensor(X_train)
            self.lidar = torch.FloatTensor(X_train_2)
            self.labels = torch.LongTensor(train_labels - 1)
        def __getitem__(self, index):
            return self.hsi[index], self.lidar[index], self.labels[index]
        def __len__(self):
            return self.len
    
    
    """ Testing dataset"""  
    class TestDS(torch.utils.data.Dataset):
        def __init__(self):
            self.len = samples_1.shape[0]
            self.hsi = torch.FloatTensor(X_test)
            self.lidar = torch.FloatTensor(X_test_2)
        def __getitem__(self, index):
            return self.hsi[index], self.lidar[index]
        def __len__(self):
            return self.len

   
    # generate trainloader and valloader
    trainset = TrainDS() 
    testset  = TestDS()
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size =X_test.shape[0], shuffle = False, num_workers = 0)



    # 6. Running
    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin
    
        def forward(self, output1, output2, label):
            euclidean_distance = F.pairwise_distance(output1, output2)
            loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2)+(1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
            return loss_contrastive
    
    
    def calc_loss(x1, x2, outputs, labels, alpha):
    
        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(outputs, labels)
    
        contrastive = ContrastiveLoss()
        loss2 = contrastive(x1, x2, labels)
    
        loss_sum = loss1 + alpha* loss2
        return loss_sum
    
    
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        total_loss = 0
        checkLoss = 50
        running_loss = 0.0
        for i, (inputs_1, inputs_2, labels) in enumerate(train_loader):
    
            inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad() 
            feat_1, feat_2, outputs = model(inputs_1, inputs_2)
            loss = calc_loss(feat_1, feat_2, outputs, labels, alpha = 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            running_loss += loss.item()
            if i % checkLoss == checkLoss-1:   #i=99，199，299.
                print('[%d %d] loss: %.3f' % (epoch+1, i+1, running_loss/checkLoss))   
                running_loss = 0.0 
    
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
    
    
    def test(model, device, test_loader):
        model.eval()
        count = 0
        for inputs_1, inputs_2 in test_loader:
    
            inputs_1, inputs_2 = inputs_1.to(device), inputs_2.to(device)
            with torch.no_grad(): # 在测试的时候必须加上这行和下一行代码，否则预测会出问题，这里是防止还有梯度更新这些，而且如果不加这个，后面又没有进行梯度更新的话，可能会报显存不够用的错误，我怀疑是数据没有被清理
                model.eval() # 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
                _, _, outputs = model(inputs_1, inputs_2)
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

        return outputs


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    
    lr = 0.01
    momentum = 0.9
    betas = (0.9, 0.999)
    
    params_to_update = list(model.parameters())
    
    # optimizer = torch.optim.Adam(params_to_update, lr=lr, betas=betas)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=0.0005)
    
    from tqdm import tqdm
    best_acc = 0
    for epoch in tqdm(range(train_epoch)):
        train(model, device, train_loader, optimizer, epoch)
        
    xxx = test(model, device, test_loader)
    
    
    
    
#%% 输出分析    
    end = datetime.datetime.now()
    CDMap = np.reshape(xxx, [im3.shape[0], im3.shape[1]])
    CDMap = np.uint8(CDMap*255)
    
    plt.imshow(CDMap, cmap='gray')
    plt.show()
    

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
    
    imsave('./results/3SAFNet/' + today + file1 + '_0255.bmp', CDMap)
    # imsave('./results/proposed/' + today + file2 + '_0255.bmp', CDMap1)
    
    f = open('./results/3SAFNet/' + file1 + '.txt', 'a')
    f.write('\n'+'Start Time: '+ str(start))
    f.write('\n'+'batch_size: '+ str(batch_size))
    f.write('\n' + today + file1 + ' TN: '+ str(zTN) + ' TP: '+ str(zTP) + ' FN: '+ str(zFN)+' FP: '+ str(zFP)+' OA: '+ str(zOA)+' Kappa: '+ str(zKappa)+' AUC: '+ str(zAUC)+' F1: '+ str(zF1_Score))
    # f.write('\n' + today + file2 +' FN1:'+ str(zzz3FN1)+' FP1: '+ str(zzz4FP1)+' OE1: '+ str(zzz5OE1)+' PCC1: '+ str(zzz6PCC1)+' KC1: '+ str(zzz8KAPPA1))
    f.write('\n')
    f.write('Time: {} '.format((end - start)))  # 时间
    f.write('\n')
    f.close()   

   
    
