# -*- coding: utf-8 -*-
"""
@author: Wenny

降维到30时， 
    班级人脸库的识别准确率为60.54%       (N = 87)   
    ORL库的识别准确率为97.5%            (N = 40)
    
运行此程序可生成文件PCA_data.mat 提供给UI文件作为数据来源
ORL库和班级库之间切换需要修改：
    N
    image_width, image_height
    faces_path
    names
    train_label, test_label
"""

import numpy as np
import cv2
import scipy.io as sio
from ctypes import c_uint8 as uint8
from pylab import mpl


mpl.rcParams['font.sans-serif'] = ['SimHei'] # 有这个画图的时候才能显示中文
N = 87 ## 人的数量
k = 10
image_width = 120
image_height = 160

# In[0]:
# 读入名字数据

def get_names():
    names = []
    with open('name.txt', 'r',encoding='utf-8') as f:
        names = f.readlines()
    for i in range(len(names)):
        names[i] = names[i].rstrip('\n')
    return names

# In[1]:
# 每个图片样本是矩阵，pca处理的样本是向量，因此需要将图片转换为向量

def img2vector(image):
    img = cv2.imread(image,0)               # 读取图片转换为灰度图片
    rows, cols = image_height, image_width  
    img = cv2.resize(img,(cols,rows))       # 将图片转换为同等大小
    imgVector = np.zeros(rows * cols)
    imgVector = np.reshape(img, (1, rows * cols)) # 使用imgVector变量作为一个向量存储图片矢量化信息，初始值均设置为0
    return imgVector

# In[2]:
# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集

def load_face(names, k): # 参数K代表选择K张图片作为训练图片使用
    '''
    对训练数据集进行数组初始化，用0填充，每张图片尺寸都定为image_height * image_width,
    现在共有N个人，每个人都选择k张，则整个训练集大小为N * k, image_height * image_width
    '''
    image_path=[]
    train_face = np.zeros((N * k, image_height * image_width), dtype=uint8)
    train_label = []
    test_face = np.zeros((N * (10 - k), image_height * image_width), dtype=uint8)
    test_label = []
    np.random.seed(0)
    sample = np.random.rand(10).argsort() + 1 # 随机排序1-10 (0-9）+1
    for i in range(N):  # 共有N个人
        for j in range(10):  # 每个人都有10张照片 
            # image = faces_path + '/s'+ str(i+1) + '/' + str(sample[j]) + '.pgm' ###ORL人脸路径
            image = faces_path + '/' + names[i] + '/' + names[i] + str(sample[j]) + '.jpg' # 同学们的人脸

            # 读取图片并进行向量化            
            img = img2vector(image)
            
            if j < k:
                # 构成训练集
                train_face[i * k + j, :] = img
                train_label.append(names[i])
                # train_label.append(i+1)
            else:
                # 构成测试集
                test_face[i * (10 - k) + (j - k), :] = img
                test_label.append(names[i])
                # test_label.append(i+1)

    return train_face, train_label, test_face, test_label, image_path

# In[3]:
# 定义PCA算法

def PCA(data, r):# 降低到r维
    data = np.double(np.mat(data))
    rows = data.shape[0]
    
    data_mean = np.mean(data, 0)                # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))    # A：差值矩阵 将所有样例减去对应均值得到A 复制到原来的rows行，1列 
    C = A.dot(A.T)                              # 协方差矩阵C = A.T * A / rows, 考虑其维数较大，根据奇异值分解定理，通过求A * A.T的特征值和特征向量来获得A.T * A的特征值和特征向量, (N*k, N*k) 
    D, V = np.linalg.eig(C)                     # 求协方差矩阵的特征值D和特征向量 V(N * k, N * k) 
    V_r = V[:, 0:r]                             # 按列取前r个特征向量 V_r(N * k, r) 
    V_r = A.T.dot(V_r)                          # 小矩阵（A * A.T）特征向量向大矩阵（A.T * A / rows）特征向量过渡 V_r(image_height * image_width, r)
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  # 特征向量归一化

    final_data = A.dot(V_r)                     # 训练脸投影到特征脸空间

    return final_data, data_mean, V_r, V

def show_face(face, data_mean, V_r): # 显示人脸照片 face为降维过后的人脸向量
    face = V_r.dot(face.reshape(-1, 1))
    face = face + data_mean.reshape(-1, 1)
    cv2.imshow("投影过后的脸", np.uint8(face.reshape(image_height, image_width)))
    cv2.waitKey(0)

# In[4]:
# 测试整个测试集的精准度


print("——————————————————提取文件———————————————————")
# 获取名字数据
names = get_names()
# names =[] # 使用ORL库时用不到names

# 获取人脸数据
# faces_path = "ORL_faces-master/ORL_faces-master/data_faces" # ORL人脸库
faces_path = "crop" # 同学们的人脸
train_face, train_label, test_face, test_label, image_path = load_face(names, k)  

print("——————————————————PCA降维————————————————————")
# 可以分别设置r(降到多少维，即选取前r个主成分)
# 经过测试，对当前数据集r=34时识别准确率最高
r = 30

# 利用PCA算法将训练样本降到r维
# data_train_new：训练集投影到特征脸空间后的数据   data_mean：训练集的平均脸   V_r：前r个特征向量   V：所有的特征向量
data_train_new, data_mean, V_r, V = PCA(train_face, r)


# 为了观察r维的特征脸空间保存原数据集特征的情况，展示降维后的其中一张照片
# 若只想生成数据可以将该语句注释掉
# show_face(data_train_new[6, :], data_mean, V_r)

# 保存数据，以便给以后测试时使用
sio.savemat("PCA_data.mat", {"train_face":train_face, "data_train_new": data_train_new, "data_mean": data_mean, "V_r":V_r, "train_label":train_label})   

print("—————————————————计算准确率——————————————————")
# 测试准确度
num_train = N * k       # 训练脸数量
num_test = N * (10 - k) # 测试脸数量
true_num = 0            # 识别成功的脸的数量
result = []             # 用于保存识别结果

# 利用训练集的矩阵变换文件，将测试集样本进行pca降维
temp_face = test_face - np.tile(data_mean, (num_test, 1))
data_test_new = temp_face.dot(V_r)  # 将待识别的人脸图像投影到特征空间，得到测试脸在特征向量下的数据

# 将得到的数据集转换为np数组，以便于后续处理
data_test_new = np.array(data_test_new)  # mat change to array
data_train_new = np.array(data_train_new)


for i in range(num_test):
    # 取一张测试脸
    testFace = data_test_new[i, :]

    # 计算测试脸到每一张训练脸的距离
    diffMat = data_train_new - np.tile(testFace, (num_train, 1))    # diffMat：测试脸到每一张训练脸的差的矩阵
    sqDiffMat = diffMat ** 2                                        # sqDiffMat：测试脸到每一张训练脸的差的平方
    sqDistances = sqDiffMat.sum(axis=1)                             # 采用欧式的是欧式距离

    sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
    indexMin = sortedDistIndicies[0]            # 距离最近的索引
    result.append(train_label[indexMin])        # 保存识别的结果
    
    if train_label[indexMin] == test_label[i]:  # 判断识别结果是否正确
        true_num += 1
    else:
        pass

# 输出准确率 
if num_test > 0:
    accuracy = float(true_num) / num_test
    print("当维数降到 %d 时,测试集人脸识别精度为: %.2f%%" % (r, accuracy * 100))




'''
for i in range(10, 20):
    cv2.imshow("平均脸", np.uint8(train_face[i, :].reshape(image_height, image_width))) # 显示晶晶
    cv2.waitKey(0)
'''