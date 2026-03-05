# =============================================================================
# args.epochs
# epochs 大概50代
# =============================================================================
from tensorflow.python.keras import layers, models, optimizers, regularizers, constraints
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras._impl.keras.layers.merge import Concatenate
# from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras._impl.keras.layers.merge import add
# from tensorflow.python.keras.layers.merge import add
import scipy.io as scio

from capsulelayer_keras import Class_Capsule, Conv_Capsule, PrimaryCap1, PrimaryCap2, Length, ecanet_layer, AFC_layer
from data_prepare import readdata



def Ms_CapsNet(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)
	#  feature extraction by AFC
    out_afc = AFC_layer(x)
    #  dim_vector is the dimensions of capsules, n_channels is number of feature maps
    Primary_caps1 = PrimaryCap1(out_afc, dim_vector=8, n_channels=4, kernel_size=1, strides=1, padding='VALID')
    # print(Primary_caps1.shape)
    Primary_caps2 = PrimaryCap2(out_afc, dim_vector=8, n_channels=4, kernel_size=1, strides=1, padding='VALID')
    # print(Primary_caps2.shape)
    
    Conv_caps1 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=args.batch_size, name='Conv_caps1')(Primary_caps1)
    Conv_caps2 = Conv_Capsule(kernel_shape=[3, 3, 4, 8], dim_vector=8, strides=[1, 2, 2, 1],
                              num_routing=num_routing, batchsize=args.batch_size, name='Conv_caps2')(Primary_caps2)
							  						
    Class_caps1 = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='class_caps1')(Conv_caps1)
    Class_caps2 = Class_Capsule(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='class_caps2')(Conv_caps2)
	#  fuse the output of class capsule  
    Class_caps_add=add([Class_caps1, Class_caps2]);
	
    out_caps = Length(name='out_caps')(Class_caps_add)

    return models.Model(x, out_caps)


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):

    (x_train, y_train), (x_valid, y_valid) = data

    # callbacks and save the training model
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/' + file1 + 'weights-test.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss],
                  metrics={' ': 'accuracy'})

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[x_valid, y_valid], callbacks=[tb, checkpoint], verbose=2)

    return model


def test(model, data): # 其实用这个也可以
    from sklearn.metrics import confusion_matrix
    x_test = data
    y_pred = model.predict(x_test, batch_size=x_test.shape[0])
    ypred = np.argmax(y_pred, 1)
    return ypred


def test1(model, data): # 原版，加了addsamples怪得很
    from sklearn.metrics import confusion_matrix
    x_test, y_test = data[0], data[1]
    n_samples = y_test.shape[0]
    add_samples = args.batch_size - n_samples % args.batch_size
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    ypred = np.argmax(y_pred, 1)
    y = np.argmax(y_test, 1)
    matrix = confusion_matrix(y[add_samples:], ypred[add_samples:])
    return matrix, ypred, add_samples


def test2(model, data):
    from sklearn.metrics import confusion_matrix
    x_test, y_test = data[0], data[1]
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    ypred = np.argmax(y_pred, 1)
    y = np.argmax(y_test, 1)
    matrix = confusion_matrix(y, ypred)
    return matrix, ypred


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA




# %% **数据修改
if __name__ == "__main__":
    import numpy as np
    import os
    # from keras import callbacks
    from tensorflow.python.keras import callbacks

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_class', default=2, type=int)  # number of classes
    parser.add_argument('--epochs', default=100, type=int) 
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--save_dir', default='./results/4CapsNet')
    parser.add_argument('--is_training', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)  # learning rate
    parser.add_argument('--windowsize', default=3, type=int) # patch size
    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


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

    file1, file2 = data10, data6
# =============================================================================
#     Yellow River I-SAR and Ottawa-SAR
# =============================================================================
    mat_file1 = io.loadmat('./images/'+file1+'.mat')
    mat_file2 = io.loadmat('./images/'+file2+'.mat')
    
    im1, im2, im3, im4, im5, im6 = mat_file1['im1'], mat_file1['im2'], mat_file1['gt_01'], \
        mat_file2['im1'], mat_file2['im2'], mat_file2['gt_01']
    
    t_samples, t_labels = mat_file1['aug_train_samples_18'], mat_file1['aug_train_labels_01'].reshape(-1)
    t_samples1, t_labels1 = mat_file2['aug_train_samples_18'], mat_file2['aug_train_labels_01'].reshape(-1)
    
    # plt.imshow(im3, cmap='gray')
    # plt.show()

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
    
    
    
    
# %% 训练
    from tensorflow.python.keras.utils import to_categorical
    x_train, y_train = result, to_categorical(t_labels)
    x_train, y_train = x_train[0:x_train.shape[0]//100*100,:,:,:], y_train[0:x_train.shape[0]//100*100,:]
    
    zzzzzz = x_train.copy()
    zzzzzz = zzzzzz[:,:,:,0]

    # define model
    model = Ms_CapsNet(input_shape=[args.windowsize, args.windowsize, 3],
                    n_class=args.n_class,
                    num_routing=args.num_routing)			
	
    train(model=model, data=((x_train, y_train), (x_train, y_train)), args=args)
    
    
    
    
# %% 测试
    print("-"*8, "train over", "-"*8)
    # model testing 
    model.load_weights(args.save_dir + '/' + file1 + 'weights-test.h5')
    
# =============================================================================
#     # 用xtrain测试有没有结果
# =============================================================================   
    # from tqdm import tqdm
    # test_nsamples = 0
    # RESULT = []
    # matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    # for i in tqdm(range(0, int(x_train.shape[0]/100))):
    #     data = (x_train[i*100:(i+1)*100,:,:,:], y_train[i*100:(i+1)*100,:])
    #     test_nsamples += data[0].shape[0]
    #     matrix1, ypred, add_samples = test1(model=model, data=(data[0], data[1]))
    #     matrix = matrix1 + matrix
    #     #matrix = matrix + test(model=model, data=(data[0], data[1]))
    #     RESULT = np.concatenate((RESULT, ypred[add_samples:]),axis = 0)
    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    # print("\n")
    # print('-' * 50)
    # print('OA:', OA)
    # print('AA:', AA_mean)
    # print('Kappa:', Kappa)
    # print('Classwise_acc:', AA)

    
# =============================================================================
#     # 用所有数据12375取12300，舍弃后面测试有没有结果
# =============================================================================
    # samples_1, srow, scol = extractPixelSamples(im1)
    # samples_2, _, _ = extractPixelSamples(im2)
    # difference = abs(samples_2/255 - samples_1/255)  # 得到形状为(m,9)的一维数组m,实际是di
    # difference = difference.reshape(-1, 3, 3)  # 转换为(m, 3，3)的二维数组
    
    # # 3. 复制三遍变成(m, 3, 3, 3)
    # expanded = difference[..., np.newaxis]   # 形状变为 (m, 4, 4, 1)
    # all_data = np.tile(expanded, (1, 1, 1, 3))   # 使用expand高效复制
    
    # ref = im3 # zz4为真相图的0/1值图
    # ref = np.array(ref,'uint8')
    # m, n = ref.shape  
    # margin = 1 
    # ref = ref[margin:m-margin, margin:n-margin]
    # ref = ref.reshape(-1)
    
    # x_train, y_train = all_data, to_categorical(ref)
    # x_train, y_train = x_train[0:x_train.shape[0]//100*100,:,:,:], y_train[0:x_train.shape[0]//100*100,:]
    
    # from tqdm import tqdm
    # test_nsamples = 0
    # RESULT = []
    # matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    # for i in tqdm(range(0, int(x_train.shape[0]/100))):
    #     data = (x_train[i*100:(i+1)*100,:,:,:], y_train[i*100:(i+1)*100,:])
    #     test_nsamples += data[0].shape[0]
    #     matrix1, ypred, add_samples = test1(model=model, data=(data[0], data[1]))
    #     matrix = matrix1 + matrix
    #     #matrix = matrix + test(model=model, data=(data[0], data[1]))
    #     RESULT = np.concatenate((RESULT, ypred[add_samples:]),axis = 0)
    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    # print("\n")
    # print('-' * 50)
    # print('OA:', OA)
    # print('AA:', AA_mean)
    # print('Kappa:', Kappa)
    # print('Classwise_acc:', AA)
    
    
    
    
# =============================================================================
#     # 用所有数据12375取12400，复制后面测试有没有结果
# =============================================================================   
    
    samples_1, srow, scol = extractPixelSamples(im1)
    expanded_1 = np.concatenate((samples_1, samples_1[:(samples_1.shape[0]//100*100+100-samples_1.shape[0])]), axis=0)
    samples_2, _, _ = extractPixelSamples(im2)
    expanded_2 = np.concatenate((samples_2, samples_2[:(samples_2.shape[0]//100*100+100-samples_2.shape[0])]), axis=0)
    difference = abs(expanded_2/255 - expanded_1/255)  # 得到形状为(m,9)的一维数组m,实际是di
    difference = difference.reshape(-1, 3, 3)  # 转换为(m, 3，3)的二维数组
    
    # 3. 复制三遍变成(m, 3, 3, 3)
    expanded = difference[..., np.newaxis]   # 形状变为 (m, 4, 4, 1)
    all_data = np.tile(expanded, (1, 1, 1, 3))   # 使用expand高效复制
    
    ref = im3 # zz4为真相图的0/1值图
    ref = np.array(ref,'uint8')
    m, n = ref.shape  
    # margin = 1 
    # ref = ref[margin:m-margin, margin:n-margin]
    ref = ref.reshape(-1)
    
    x_train, y_train = all_data, to_categorical(ref)
    y_train = np.concatenate((y_train, y_train[:(y_train.shape[0]//100*100+100-y_train.shape[0])]), axis=0)
    x_train, y_train = x_train[0:x_train.shape[0]//100*100,:,:,:], y_train[0:x_train.shape[0]//100*100,:]
    
    zzz = np.zeros((x_train.shape[0], 1))
    
    from tqdm import tqdm
    test_nsamples = 0
    RESULT = []
    matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    for i in tqdm(range(0, int(x_train.shape[0]/100))):
        data = (x_train[i*100:(i+1)*100,:,:,:], y_train[i*100:(i+1)*100,:])
        test_nsamples += data[0].shape[0]
        matrix1, ypred = test2(model=model, data=(data[0], data[1]))
        matrix = matrix1 + matrix
        #matrix = matrix + test(model=model, data=(data[0], data[1]))
        zzz[i*100:(i+1)*100] = ypred.reshape(-1,1)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    print("\n")
    print('-' * 50)
    print('OA:', OA)
    print('AA:', AA_mean)
    print('Kappa:', Kappa)
    print('Classwise_acc:', AA)


    xxx = zzz[0:samples_1.shape[0], :].reshape(-1)
    end = datetime.datetime.now()
    CDMap = np.reshape(xxx, [im3.shape[0], im3.shape[1]])
    CDMap = np.uint8(CDMap*255)
    
    # plt.imshow(CDMap, cmap='gray')
    # plt.show()
    
#%% **输出分析 
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
    
    imsave('./results/4CapsNet/' + today + file1 + '_0255.bmp', CDMap)
    # imsave('./results/proposed/' + today + file2 + '_0255.bmp', CDMap1)
    
    f = open('./results/4CapsNet/' + file1 + '.txt', 'a')
    f.write('\n'+'Start Time: '+ str(start))
    f.write('\n'+'batch_size: '+ str(args.batch_size))
    f.write('\n' + today + file1 + ' TN: '+ str(zTN) + ' TP: '+ str(zTP) + ' FN: '+ str(zFN)+' FP: '+ str(zFP)+' OA: '+ str(zOA)+' Kappa: '+ str(zKappa)+' AUC: '+ str(zAUC)+' F1: '+ str(zF1_Score))
    # f.write('\n' + today + file2 +' FN1:'+ str(zzz3FN1)+' FP1: '+ str(zzz4FP1)+' OE1: '+ str(zzz5OE1)+' PCC1: '+ str(zzz6PCC1)+' KC1: '+ str(zzz8KAPPA1))
    f.write('\n')
    f.write('Time: {} '.format((end - start)))  # 时间
    f.write('\n')
    f.close()   

    

        
    

