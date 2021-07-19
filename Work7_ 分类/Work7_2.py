# 根据散点图分析，房屋的’RM（每栋住宅的房间数）’，‘LSTAT（地区中有多少房东属于低收入人群）’，特征与房价的相关性最大
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PrtalWIdth', 'Species']
tf_iris = pd.read_csv(train_path, names=COLUMN_NAMES, header=0)
COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PrtalWIdth', 'Species']
tf_iris_test = pd.read_csv(test_path, names=COLUMN_NAMES, header=0)
iris = np.array(tf_iris)  # 下载数据集
iris_test = np.array(tf_iris_test) # 下载测试集

# ---加载数据
iris = iris[iris[:,4] > 0]

x_train_2 = iris[:, 0:2]
x_train_3 = iris[:, 0:3]
x_train_4 = iris[:, 0:4]
y_train = iris[:, 4]

num = len(y_train)   # 属性长度

# 属性中心化
x_train_2 = x_train_2 - tf.reduce_mean(x_train_2, axis=0)
x_train_3 = x_train_3 - tf.reduce_mean(x_train_3, axis=0)
x_train_4 = x_train_4 - tf.reduce_mean(x_train_4, axis=0)
y_train = y_train.reshape(-1,1)
# print(len(x_train))
# print(type(y_train))

# ---数据处理/线性归一化
x0_train = np.ones(num).reshape(-1,1)

X_train_2 = tf.cast(tf.concat((x0_train,x_train_2), axis=1), tf.float32)#.reshape(-1, 1)
X_train_3 = tf.cast(tf.concat((x0_train,x_train_3), axis=1), tf.float32)#.reshape(-1, 1)
X_train_4 = tf.cast(tf.concat((x0_train,x_train_4), axis=1), tf.float32)#.reshape(-1, 1)
Y_train = tf.cast(tf.where(y_train==2,0,1), tf.float32)

# ---设置超参数
learn_rate = 0.5
iter = 50
display_step = 10
# ---设置模型参数初值w
np.random.seed(612)
w_2 = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)
w_3 = tf.Variable(np.random.randn(4, 1), dtype=tf.float32)
w_4 = tf.Variable(np.random.randn(5, 1), dtype=tf.float32)
# ---训练模型
ce_2 = []  # 每次迭代的交叉熵损失
acc_2 = []  # 准确率
ce_3 = []  # 每次迭代的交叉熵损失
acc_3 = []  # 准确率
ce_4 = []  # 每次迭代的交叉熵损失
acc_4 = []  # 准确率

# ce 
# def classify(*ce, *acc, X_train, w, att_num):
#     for i in range(0, iter + 1):
#         with tf.GradientTape(persistent=True) as tape:
#         pred = 1 / (1 + tf.exp(-tf.matmul(X_train, w)))  # sigmoid函数 每个样本的预测概率
#         Loss = -tf.reduce_mean(Y_train * tf.math.log(pred) + (1 - Y_train) * tf.math.log(1 - pred))  # 平均交叉熵损失
#         dL_dw = tape.gradient(Loss, w)
#         w.assign_sub(learn_rate*dL_dw)
#         accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred.numpy() < 0.5, 0., 1.), Y_train), tf.float32))
#         ce.append(Loss)
#         acc.append(accuracy)
#     if i % display_step == 0:
#         print("i:%i, Acc:%f, Loss:%f" % (i, accuracy, Loss))

# classify(*ce_2, *acc_2, X_train_2, w_2, 2)

for i in range(0, iter + 1):
    with tf.GradientTape(persistent=True) as tape:
        pred_2 = 1 / (1 + tf.exp(-tf.matmul(X_train_2, w_2)))  # sigmoid函数 每个样本的预测概率
        Loss_2 = -tf.reduce_mean(Y_train * tf.math.log(pred_2) + (1 - Y_train) * tf.math.log(1 - pred_2))  # 平均交叉熵损失
        pred_3 = 1 / (1 + tf.exp(-tf.matmul(X_train_3, w_3)))  # sigmoid函数 每个样本的预测概率
        Loss_3 = -tf.reduce_mean(Y_train * tf.math.log(pred_3) + (1 - Y_train) * tf.math.log(1 - pred_3))  # 平均交叉熵损失
        pred_4 = 1 / (1 + tf.exp(-tf.matmul(X_train_4, w_4)))  # sigmoid函数 每个样本的预测概率
        Loss_4 = -tf.reduce_mean(Y_train * tf.math.log(pred_4) + (1 - Y_train) * tf.math.log(1 - pred_4))  # 平均交叉熵损失

    dL2_dw = tape.gradient(Loss_2, w_2)
    w_2.assign_sub(learn_rate*dL2_dw)
    accuracy_2 = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_2.numpy() < 0.5, 0., 1.), Y_train), tf.float32))
    dL3_dw = tape.gradient(Loss_3, w_3)
    w_3.assign_sub(learn_rate*dL3_dw)
    accuracy_3 = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_3.numpy() < 0.5, 0., 1.), Y_train), tf.float32))
    dL4_dw = tape.gradient(Loss_4, w_4)
    w_4.assign_sub(learn_rate*dL4_dw)
    accuracy_4 = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_4.numpy() < 0.5, 0., 1.), Y_train), tf.float32))

    ce_2.append(Loss_2)
    acc_2.append(accuracy_2)
    ce_3.append(Loss_3)
    acc_3.append(accuracy_3)
    ce_4.append(Loss_4)
    acc_4.append(accuracy_4)
    if i % display_step == 0:
        print("i:%i, Acc2:%f, Loss2:%f\tAcc3:%f, Loss3:%f\tAcc4:%f, Loss4:%f" % (i, accuracy_2, Loss_2, accuracy_3, Loss_3, accuracy_4, Loss_4))

plt.figure(figsize=(5,3))
plt.subplot(131)
plt.plot(ce_2, c='b', label='Loss2')
plt.plot(acc_2, c='r', label='acc2')
plt.subplot(132)
plt.plot(ce_3, c='g', label='Loss3')
plt.plot(acc_3, c='r', label='acc3')
plt.subplot(133)
plt.plot(ce_4, c='y', label='Loss4')
plt.plot(acc_4, c='r', label='acc4')
plt.legend()

plt.show()

cm_pt = mpl.colors.ListedColormap(["blue", "red"])
plt.scatter(X_train_2[:,1], X_train_2[:,2], c=y_train, cmap=cm_pt)
x_ = [-1.5, 1.5]
y_ = -(w_2[1]*x_ + w_2[0])/w_2[2]
plt.plot(x_, y_, c='g')

plt.show()

from mpl_toolkits.mplot3d import Axes3D

X1,X2 = np.meshgrid(X_train_3[:,1], X_train_3[:,2])
Y_PRED = w_3[1]*X1+w_3[2]*X2+w_3[0]
fig = plt.figure(figsize = (8,6))
ax3d = Axes3D(fig)
ax3d.scatter(X_train_3[:,1], X_train_3[:,2], X_train_3[:,3], c=y_train, cmap=cm_pt)
ax3d.plot_surface(X1,X2,Y_PRED,cmap="Pastel1")

plt.show()