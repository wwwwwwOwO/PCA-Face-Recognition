import time
import sys
import cv2
import scipy.io as sio
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *

import Ui_Form
Ui_Form = Ui_Form.Ui_Form

N = 87 #
k = 10
image_width = 120 #
image_height = 160 #

# 加载之前保存过的数据
PCA_data = sio.loadmat("PCA_data.mat")
train_face = PCA_data["train_face"]
data_train_new = PCA_data["data_train_new"]
data_mean = PCA_data["data_mean"]
V_r = PCA_data["V_r"]
train_label = PCA_data["train_label"]
num_train = N * k

class CoperQt(QtWidgets.QMainWindow,Ui_Form):#创建一个Qt对象

    def __init__(self):
        self.timer_camera = QtCore.QTimer() # 定时器
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # 准备获取图像 打开默认摄像头（序号0）
        
        QtWidgets.QMainWindow.__init__(self)    # 创建主界面对象
        Ui_Form.__init__(self)                  # 主界面对象初始化
        self.setupUi(self)                      # 配置主界面对象

        # 匹配槽
        self.timer_camera.timeout.connect(self.showCamera)  # 显示摄像头画面
        self.pushButton_2.clicked.connect(self.openCamera)  # 打开摄像头
        self.pushButton_3.clicked.connect(self.shoot)       # 拍摄照片
        self.pushButton_4.clicked.connect(self.closeCamera) # 关闭摄像头
        self.lineEdit.returnPressed.connect(self.recognize) # 输入文件时回车开始识别
        self.pushButton.clicked.connect(self.recognize)     # 点击确定开始识别
        self.toolButton.clicked.connect(self.openFile)      # 在文件资源管理器中选择人脸文件
    
    def showImage(self, label, img):
        img = img.copy()    # 不知道为什么，不加这句train_face里的人脸就显示不出来，查了很多资料都没有解答
        QIm = QImage(img.data, image_width, image_height, image_width, QImage.Format_Indexed8)    # 创建QImage格式的图像，并读入图像信息
        
        # label.setPixmap(QPixmap.fromImage(QIm).scaled(
        #     self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))  
        label.setPixmap(QPixmap.fromImage(QIm))  

    def recognize(self):
        path = self.lineEdit.text()                 # 获取输入的文件路径
        path = eval(repr(path).replace('\\', '/'))  # 把左斜杠'\'转为右斜杠'/'

        testFace = cv2.imread(path, 0)                                  # 读取图片转换为灰度图片
        testFace = cv2.resize(testFace, (image_width, image_height))    # 转化为标准大小
        self.showImage(self.label_5, testFace)                          # 展示原图
        
        # 测试脸预处理，映射到特征脸空间
        testFace = np.reshape(testFace, (1, image_height * image_width))
        testFace = testFace - data_mean
        testFace = testFace.dot(V_r)

        # 计算测试脸到每一张训练脸的距离
        diffMat = data_train_new - np.tile(testFace, (num_train, 1))    # diffMat：测试脸到每一张训练脸的差的矩阵
        sqDiffMat = diffMat ** 2                                        # sqDiffMat：测试脸到每一张训练脸的差的平方
        sqDistances = sqDiffMat.sum(axis=1)                             # 采用欧式的是欧式距离
        sortedDistIndicies = sqDistances.argsort()                      # 对向量从小到大排序，使用的是索引值,得到一个向量
        indexMin = sortedDistIndicies[0]                                # 距离最近的索引

        # self.label_12.setText("s"+str(train_label[0, indexMin]))        # 输出最匹配的照片对应的人名 ORL
        self.label_12.setText(train_label[indexMin])                      # 输出最匹配的照片对应的人名

        # 展示训练集内最相似的5张人脸
        face = train_face[sortedDistIndicies[0], :]
        self.showImage(self.label_10, face.reshape(image_height, image_width))
        face = train_face[sortedDistIndicies[1], :]
        self.showImage(self.label_9 , face.reshape(image_height, image_width))
        face = train_face[sortedDistIndicies[2], :]
        self.showImage(self.label_7 , face.reshape(image_height, image_width))
        face = train_face[sortedDistIndicies[3], :]
        self.showImage(self.label_8 , face.reshape(image_height, image_width))
        face = train_face[sortedDistIndicies[4], :]
        self.showImage(self.label_6 , face.reshape(image_height, image_width))

    def openFile(self):
        # 获取文件路径对话框 "文件资源管理器"为文件对话框的标题，第三个是打开的默认路径，第四个是文件类型过滤器，只读取jpg和pgm格式的文件
        file_name = QFileDialog.getOpenFileName(self,"文件资源管理器","D:\Python practise\practise\PCA\crop","JPG files(*.jpg);;PGM Files(*.pgm)") 
        file_name = str(file_name[0])       # 获取文件名file_name的第一个元素为文件完整路径，第二个元素为文件类型
        self.lineEdit.setText(file_name)    # 写入文件路径


    def showCamera(self):
        flag, self.image = self.cap.read()
        # print(self.image.shape) ------>(480, 640, 3)
        self.image = self.image[0:480, 140:500, :] # 截取中间比例为(4, 3)的部分，不然resize后照片会被拉伸，人脸会很长 --->shape = (480, 360, 3)
        show = cv2.resize(self.image,(image_width, image_height))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        self.label_4.setPixmap(QPixmap.fromImage(showImage))

    def openCamera(self):
        self.timer_camera.start(70)     # 开启计数器
            
    def closeCamera(self):
        self.timer_camera.stop()        # 关闭计数器
        self.label_4.clear()            # 清屏

    def shoot(self):
        self.timer_camera.stop()                    # 暂停计数
        flag, self.image = self.cap.read()          # 获取当前画面
        self.image = self.image[0:480, 140:500, :]  # 取中央4:3的部分
        cv2.imwrite("shooting.jpg", self.image)     # 保存图片为jpg形式
        self.lineEdit.setText("shooting.jpg")       # 写入当前文件名到待测试区
        time.sleep(1)                               # 延时1s
        self.timer_camera.start(70)                 # 开启计数

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CoperQt()  # 创建QT对象
    window.show()       # QT对象显示
    sys.exit(app.exec_())