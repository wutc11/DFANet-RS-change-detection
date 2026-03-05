import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from dag import DAG
from tqdm import tqdm
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import evolution
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
import scipy.io as scio
from tf_utils import random_mini_batches_single, convert_to_one_hot, one_hot_back
from imageio import imread, imsave


# ##########################################################################################
def segmented_process(M, blk_size=(16,16), overlap=(0,0)):
    rows1 = []
    for i in range(0, M.shape[0]-blk_size[0]+1, blk_size[0]-overlap[0]):
        cols1 = []
        for j in range(0, M.shape[1]-blk_size[1]+1, blk_size[1]-overlap[1]):
            cols1.append(M[i:i+blk_size[0], j:j+blk_size[1]])
        rows1.append(cols1)
    rows = []  
    
    
    for i in range(len(rows1)):
        cols = []
        for j in range(len(rows1[0])):
            cols.append(np.concatenate(rows1[i][j].reshape(1,-1)))
        rows.append(np.concatenate(cols))
    rows = np.concatenate(rows)  
    
    return rows

def data_reconstruct(im1, w, over): # 图像提取邻窗
    xtrain=[] 
    for i in range(im1.shape[2]):
        im = im1[:,:,i]
        im = segmented_process(im, blk_size=(w,w), overlap=(over, over))
        im = im.reshape(-1, w*w)
        if i == 0:
            xtrain = im
        else:
            xtrain = np.append(xtrain, im, axis=1)
            
    return xtrain

def data_loadmat(d1_im1, d1_im2, d1_di, d1_real):
    im1 = data_reconstruct(d1_im1, 3, 2) # 3,2为邻窗大小和去周围值
    im2 = data_reconstruct(d1_im2, 3, 2)
    im1, im2 = im1.reshape(-1, 3, 3, 3), im2.reshape(-1, 3, 3, 3)
    X_train = np.dstack((im1, im2))
    Y_train = d1_di[:,:,0]
    row, col = Y_train.shape
    Y_train = Y_train[1:row-1, 1:col-1] # 对应了去周围值，从1到n-1的范围
    Y_train=convert_to_one_hot(Y_train.astype(int), 2)
    Y_train=Y_train.T
    X_test = X_train
    Y_test = d1_real[1:row-1, 1:col-1]
    Y_test=convert_to_one_hot(Y_test.astype(int), 2)
    Y_test=Y_test.T
    
    return X_train, Y_train, X_test, Y_test




import datetime
import time
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


# %% **数据处理
file1, file2, file3, file4 = data9, data10, data9, data10
# =============================================================================
#     Yellow River I-SAR and Ottawa-SAR
# =============================================================================
mat_file1 = io.loadmat('./images/'+file1+'.mat')
mat_file2 = io.loadmat('./images/'+file2+'.mat')
mat_file3 = io.loadmat('./images/'+file3+'.mat')
mat_file4 = io.loadmat('./images/'+file4+'.mat')


im1, im2, im3, im4, im5, im6 = mat_file1['im1'], mat_file1['im2'], mat_file1['gt_01'], \
    mat_file2['im1'], mat_file2['im2'], mat_file2['gt_01']
im7, im8, im9, im10, im11, im12 = mat_file3['im1'], mat_file3['im2'], mat_file3['gt_01'], \
    mat_file4['im1'], mat_file4['im2'], mat_file4['gt_01']

t_samples, t_labels = mat_file1['aug_train_samples_18'], mat_file1['aug_train_labels_01'].reshape(-1)
t_samples1, t_labels1 = mat_file2['aug_train_samples_18'], mat_file2['aug_train_labels_01'].reshape(-1)
t_samples2, t_labels2 = mat_file3['aug_train_samples_18'], mat_file3['aug_train_labels_01'].reshape(-1)
t_samples3, t_labels3 = mat_file4['aug_train_samples_18'], mat_file4['aug_train_labels_01'].reshape(-1)

di, di1, di2, di3 = mat_file1['di_01'], mat_file2['di_01'], mat_file3['di_01'], mat_file4['di_01']
# plt.imshow(im3, cmap='gray')
# plt.show()
# plt.imshow(di, cmap='gray')
# plt.show()
# plt.imshow(im6, cmap='gray')
# plt.show()
# plt.imshow(di1, cmap='gray')
# plt.show()
# plt.imshow(im9, cmap='gray')
# plt.show()
# plt.imshow(di2, cmap='gray')
# plt.show()
# plt.imshow(im12, cmap='gray')
# plt.show()
# plt.imshow(di3, cmap='gray')
# plt.show()

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
im7 = preprocess_image(im7)
im8 = preprocess_image(im8)
im10 = preprocess_image(im10)
im11 = preprocess_image(im11)  

from tensorflow.python.keras.utils import to_categorical
def data_loadmat1(t_samples, t_labels):
    # 1. 提取筛选的im1和im2的数据
    col4 = t_samples[:, 0:9]   
    col13 = t_samples[:, 9:18]  
    
    # 2. 分别得到im1,2
    get_im1 = col4.reshape(-1, 3, 3)   # 得到形状为(m,9)的一维数组m,实际是di
    get_im2 = col13.reshape(-1, 3, 3)  # 转换为(m, 3，3)的二维数组
    
    # 3. 拓展变成(m, 3, 3, 1)
    train_1 = get_im1[..., np.newaxis]   # 形状变为 (m, 4, 4, 1)
    train_2 = get_im2[..., np.newaxis] 
    
    expanded = np.dstack((train_1, train_2))
    X_train = np.tile(expanded, (1, 1, 1, 3))   # 使用expand高效复制
    Y_train = to_categorical(t_labels)
    return X_train, Y_train

X_train1, Y_train1 = data_loadmat1(t_samples, t_labels)
X_train2, Y_train2 = data_loadmat1(t_samples1, t_labels1)
X_train3, Y_train3 = data_loadmat1(t_samples2, t_labels2)
X_train4, Y_train4 = data_loadmat1(t_samples3, t_labels3)

def test_loadmat1(im1, im2, im3):
    # 准备所有test数据
    samples_1, srow, scol = extractPixelSamples(im1)
    samples_2, _, _ = extractPixelSamples(im2)
    X_test, X_test_2 = samples_1.reshape(-1, 3, 3), samples_2.reshape(-1, 3, 3)
    X_test, X_test_2 = X_test[..., np.newaxis], X_test_2[..., np.newaxis]
    expanded = np.dstack((X_test, X_test_2))
    X_train = np.tile(expanded, (1, 1, 1, 3))   # 使用expand高效复制
    
    ref = im3 * 255 # zz4为真相图的0/1值图
    ref = np.array(ref,'uint8')
    m, n = ref.shape  
    # margin = 1 
    # ref = ref[margin:m-margin, margin:n-margin]
    gt = ref.copy()
    ref = ref.reshape(-1)
    ref = ref/255
    Y_train = to_categorical(ref)

    return X_train, Y_train, gt

X_test1, Y_test1, gt1 = test_loadmat1(im1, im2, im3)
X_test2, Y_test2, gt2 = test_loadmat1(im4, im5, im6)
X_test3, Y_test3, gt3 = test_loadmat1(im7, im8, im9)
X_test4, Y_test4, gt4 = test_loadmat1(im10, im11, im12)    




# %% 网络训练
STAGES = np.array(["s1", "s2"])
NUM_NODES = np.array([2, 5])
# print(NUM_NODES)
L = 0
BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0  # to keep track of bits for each stage S
for nn in NUM_NODES:
    t = nn * (nn - 1)
    BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
    l_bpi = int(0.5 * t)
    L += t
L = int(0.5 * L)

"!!!!这边训练次数也改小了"
TRAINING_EPOCHS = 50
BATCH_SIZE = 128
population_size = 10
num_generations = 1

def weight_variable(weight_name, weight_shape):
    return tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name=''.join(["weight_", weight_name]))


def bias_variable(bias_name, bias_shape):
    return tf.Variable(tf.constant(0.01, shape=bias_shape), name=''.join(["bias_", bias_name]))


def linear_layer(x, n_hidden_units, layer_name):
    n_input = int(x.get_shape()[1])
    weights = weight_variable(layer_name, [n_input, n_hidden_units])
    biases = bias_variable(layer_name, [n_hidden_units])
    return tf.add(tf.matmul(x, weights), biases)


def apply_convolution(x, kernel_height, kernel_width, in_channels, out_chanels, layer_name):
    weights = weight_variable(layer_name, [kernel_height, kernel_width, in_channels, out_chanels])
    biases = bias_variable(layer_name, [out_chanels])
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 2, 2, 1], padding="SAME"), biases))


def apply_pool(x, kernel_height, kernel_width, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, stride_size, stride_size, 1], padding="SAME")


def add_node(node_name, connector_node_name, h=5, w=5, ic=1, oc=1):
    with tf.name_scope(node_name) as scope:
        conv = apply_convolution(tf.get_default_graph().get_tensor_by_name(connector_node_name),
                                 kernel_height=h, kernel_width=w, in_channels=ic, out_chanels=oc,
                                 layer_name=''.join(["conv_", node_name]))


def sum_tensors(tensor_a, tensor_b, activation_function_pattern):
    if not tensor_a.startswith("Add"):
        tensor_a = ''.join([tensor_a, activation_function_pattern])

    return tf.add(tf.get_default_graph().get_tensor_by_name(tensor_a),
                  tf.get_default_graph().get_tensor_by_name(''.join([tensor_b, activation_function_pattern])))


def has_same_elements(x):
    return len(set(x)) <= 1


'''This method will come handy to first generate DAG independent of Tensorflow, 
    afterwards generated graph can be used to generate Tensorflow graph'''


def generate_dag(optimal_indvidual, stage_name, num_nodes):
    # create nodes for the graph
    nodes = np.empty((0), dtype=np.str)
    for n in range(1, (num_nodes + 1)):
        nodes = np.append(nodes, ''.join([stage_name, "_", str(n)]))

    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    # split best indvidual found via GA to identify vertices connections 
    edges = np.split(optimal_indvidual, np.cumsum(range(num_nodes - 1)))[1:]
    v2 = 2
    for e in edges:
        v1 = 1
        for i in e:
            if i:
                dag.add_edge(''.join([stage_name, "_", str(v1)]), ''.join([stage_name, "_", str(v2)]))
            v1 += 1
        v2 += 1

    # delete nodes not connected to anyother node
    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])

    return dag, nodes


def generate_tensorflow_graph(individual, stages, num_nodes, bits_indices,X_train,Y_train):
    activation_function_pattern = "/Relu:0"

    tf.reset_default_graph()
    
    # tf.disable_eager_execution()
 

    # print(X_train.shape)
    # print(Y_train.shape)
    X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1],X_train.shape[2],X_train.shape[3]], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name="Y")

    d_node = X
    # print("d_node",d_node)
    for stage_index, stage_name, num_node, bpi in zip(range(0, len(stages)),stages, num_nodes, bits_indices):
        indv = individual[bpi[0]:bpi[1]]

        ic = 1
        oc = 1
        "!!!!ic为一开始的输入通道，oc为输出通道，input channel"
        if stage_index == 0: #"!!!!ic为一开始的输入通道，oc为输出通道，input channel"
            add_node(''.join([stage_name, "_input"]), d_node.name, ic=3, oc=20) # ic=3
            ic = 20
            oc = 20
        elif stage_index == 1:
            add_node(''.join([stage_name, "_input"]), d_node.name, ic=20, oc=50)
            ic = 50
            oc = 50

        pooling_layer_name = ''.join([stage_name, "_input", activation_function_pattern])

        if not has_same_elements(indv):
            # ------------------- Temporary DAG to hold all connections implied by GA solution ------------- #

            # get DAG and nodes in the graph
            dag, nodes = generate_dag(indv, stage_name, num_node)
            # get nodes without any predecessor, these will be connected to input node
            without_predecessors = dag.ind_nodes()
            # get nodes without any successor, these will be connected to output node
            without_successors = dag.all_leaves()

            # ----------------------------------------------------------------------------------------------- #

            # --------------------------- Initialize tensforflow graph based on DAG ------------------------- #

            for wop in without_predecessors:
                add_node(wop, ''.join([stage_name, "_input", activation_function_pattern]), ic=ic, oc=oc)

            for n in nodes:
                predecessors = dag.predecessors(n)
                if len(predecessors) == 0:
                    continue
                elif len(predecessors) > 1:
                    first_predecessor = predecessors[0]
                    for prd in range(1, len(predecessors)):
                        t = sum_tensors(first_predecessor, predecessors[prd], activation_function_pattern)
                        first_predecessor = t.name
                    add_node(n, first_predecessor, ic=ic, oc=oc)
                elif predecessors:
                    add_node(n, ''.join([predecessors[0], activation_function_pattern]), ic=ic, oc=oc)

            if len(without_successors) > 1:
                first_successor = without_successors[0]
                for suc in range(1, len(without_successors)):
                    t = sum_tensors(first_successor, without_successors[suc], activation_function_pattern)
                    first_successor = t.name
                add_node(''.join([stage_name, "_output"]), first_successor, ic=ic, oc=oc)
            else:
                add_node(''.join([stage_name, "_output"]),
                         ''.join([without_successors[0], activation_function_pattern]), ic=ic, oc=oc)

            pooling_layer_name = ''.join([stage_name, "_output", activation_function_pattern])
            # ------------------------------------------------------------------------------------------ #

        d_node = apply_pool(tf.get_default_graph().get_tensor_by_name(pooling_layer_name),
                            kernel_height=2, kernel_width=2, stride_size=2)
        # print("d_node",d_node)

    shape = d_node.get_shape().as_list()
    flat = tf.reshape(d_node, [-1, shape[1] * shape[2] * shape[3]])
    logits500 = tf.nn.dropout(linear_layer(flat, 500, "logits500"), 0.5, name="dropout")
    "!!!!这里为最终的输出层，2对应了标签的维度，他之前是16类"
    logits = linear_layer(logits500, 2, "logits")
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)  #交叉熵
    loss_function = tf.reduce_mean(xentropy) #平均值
    optimizer = tf.train.AdamOptimizer().minimize(loss_function)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1)), tf.float32))

    return X, Y, optimizer, loss_function, accuracy,logits


def evaluateModel(individual, task):
    print("第", task, "个任务进行测试")
    score = 0.0
    "!!!!这里全部用train1和test1代替"
    if task == 1:
        X_train = X_train1
        X_test = X_test1
        Y_train = Y_train1
        Y_test = Y_test1
    elif task == 2:
        X_train = X_train2
        X_test = X_test2
        Y_train = Y_train2
        Y_test = Y_test2
    elif task == 3:
        X_train = X_train3
        X_test = X_test3
        Y_train = Y_train3
        Y_test = Y_test3
    else:
        X_train = X_train4
        X_test = X_test4
        Y_train = Y_train4
        Y_test = Y_test4

    X, Y, optimizer, loss_function, accuracy,logits = generate_tensorflow_graph(individual, STAGES, NUM_NODES, BITS_INDICES,X_train,Y_train)
    # print("X",X)
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for epoch in range(TRAINING_EPOCHS):
            # if epoch%5 ==0:
            #     print("epoch:", epoch)
            for b in range(Y_train.shape[0]%BATCH_SIZE):
                batch_x = X_train[b*BATCH_SIZE:(b+1)*BATCH_SIZE, :, :, :]
                batch_y = Y_train[b*BATCH_SIZE:(b+1)*BATCH_SIZE, :]
                _, c = session.run([optimizer, loss_function], feed_dict={X: batch_x, Y: batch_y})
            
            # for b in range(3):
            #     batch_x = X_train[b*BATCH_SIZE:(b+1)*BATCH_SIZE, :, :, :]
            #     batch_y = Y_train[b*BATCH_SIZE:(b+1)*BATCH_SIZE, :]
            #     _, c = session.run([optimizer, loss_function], feed_dict={X: batch_x, Y: batch_y})

        
        score = session.run([accuracy], feed_dict={X: X_test, Y: Y_test})
        logits = session.run([logits], feed_dict={X: X_test, Y: Y_test})
        print('Accuracy: ',score)
    # X, Y, optimizer, loss_function, accuracy,logits = generate_tensorflow_graph(individual, STAGES, NUM_NODES, BITS_INDICES,X_test,Y_test)
    return score,logits




# %% 结果测试
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("factorial_ranks", list)

creator.create("Individual", list, fitness=creator.FitnessMax, fitness1=creator.FitnessMax, fitness2=creator.FitnessMax,
                fitness3=creator.FitnessMax, fitness4=creator.FitnessMax,
                factorial_ranks=creator.factorial_ranks, scalar_fitness=creator.FitnessMax,
                skill_factor=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("binary", bernoulli.rvs, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=L)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.8)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluateModel)
# popl是初始化后的染色体，而它下面一句是进化过程，具体可以打开algorithm算法看一看
popl = toolbox.population(n=population_size)
pop,logbook,cd1,cd2,cd3,cd4 = evolution.eaSimple(popl, toolbox, cxpb=0.4, mutpb=0.05, ngen=num_generations, task_num=4, verbose=True)
# print(pop)
end = datetime.datetime.now()
# print("\n"+"cd1:",cd1)
# print("\n"+"cd2:",cd2)
# print("\n"+"cd3:",cd3)
# print("\n"+"cd4:",cd4)
cd1 = one_hot_back(cd1)
cd2 = one_hot_back(cd2)
cd3 = one_hot_back(cd3)
cd4 = one_hot_back(cd4)

cd1 = cd1.reshape(-1)
cd2 = cd2.reshape(-1)
cd3 = cd3.reshape(-1)
cd4 = cd4.reshape(-1)

def out_results(CDMap, gt):
    image = CDMap*255
    ref = gt

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

    return image, zTN, zFP, zFN, zTP, zOA, zKappa, zAUC, zF1_Score


#%% **输出分析
CDMap, zTN, zFP, zFN, zTP, zOA, zKappa, zAUC, zF1_Score = out_results(cd1.reshape(gt1.shape[0], gt1.shape[1]), gt1)
CDMap1, zTN1, zFP1, zFN1, zTP1, zOA1, zKappa1, zAUC1, zF1_Score1 = out_results(cd2.reshape(gt2.shape[0], gt2.shape[1]), gt2)
CDMap2, zTN2, zFP2, zFN2, zTP2, zOA2, zKappa2, zAUC2, zF1_Score2 = out_results(cd3.reshape(gt3.shape[0], gt3.shape[1]), gt3)
CDMap3, zTN3, zFP3, zFN3, zTP3, zOA3, zKappa3, zAUC3, zF1_Score3 = out_results(cd4.reshape(gt4.shape[0], gt4.shape[1]), gt4)
print("CDMap:",CDMap.shape)

imsave('./results/8EMCNNAS/' + today + file1 + '_0255.bmp', CDMap)
imsave('./results/8EMCNNAS/' + today + file2 + '_0255.bmp', CDMap1)
imsave('./results/8EMCNNAS/' + today + file3 + '_0255.bmp', CDMap2)
imsave('./results/8EMCNNAS/' + today + file4 + '_0255.bmp', CDMap3)


f = open('./results/8EMCNNAS/' + file1 + '+' + file2 + '+' + file3 + '+' + file4 + '.txt', 'a')
f.write('\n'+'Start Time: '+ str(start))
f.write('\n'+'batch_size: '+ str(BATCH_SIZE))
f.write('\n' + today + file1 + ' TN: '+ str(zTN) + ' TP: '+ str(zTP) + ' FN: '+ str(zFN)+' FP: '+ str(zFP)+' OA: '+ str(zOA)+' Kappa: '+ str(zKappa)+' AUC: '+ str(zAUC)+' F1: '+ str(zF1_Score))
f.write('\n' + today + file2 + ' TN: '+ str(zTN1) + ' TP: '+ str(zTP1) + ' FN: '+ str(zFN1)+' FP: '+ str(zFP1)+' OA: '+ str(zOA1)+' Kappa: '+ str(zKappa1)+' AUC: '+ str(zAUC1)+' F1: '+ str(zF1_Score1))
f.write('\n' + today + file3 + ' TN: '+ str(zTN2) + ' TP: '+ str(zTP2) + ' FN: '+ str(zFN2)+' FP: '+ str(zFP2)+' OA: '+ str(zOA2)+' Kappa: '+ str(zKappa2)+' AUC: '+ str(zAUC2)+' F1: '+ str(zF1_Score2))
f.write('\n' + today + file4 + ' TN: '+ str(zTN3) + ' TP: '+ str(zTP3) + ' FN: '+ str(zFN3)+' FP: '+ str(zFP3)+' OA: '+ str(zOA3)+' Kappa: '+ str(zKappa3)+' AUC: '+ str(zAUC3)+' F1: '+ str(zF1_Score3))
f.write('\n')
f.write('Time: {} '.format((end - start)))  # 时间
f.write('\n')
f.close()  

# =============================================================================
# # print top-3 optimal solutions
# best_individuals_task1 = tools.selBest(popl, k=3, fit_attr="fitness1")
# best_individuals_task2 = tools.selBest(popl, k=3, fit_attr="fitness2")
# best_individuals_task3 = tools.selBest(popl, k=3, fit_attr="fitness3")
# best_individuals_task4 = tools.selBest(popl, k=3, fit_attr="fitness4")
# 
# for bi in best_individuals_task1:
#     print(bi)
# for bi in best_individuals_task2:
#     print(bi)
# for bi in best_individuals_task3:
#     print(bi)
# for bi in best_individuals_task4:
#     print(bi)
# =============================================================================
    

