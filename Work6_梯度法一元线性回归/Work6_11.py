# 根据散点图分析，房屋的’RM（每栋住宅的房间数）’，‘LSTAT（地区中有多少房东属于低收入人群）’，特征与房价的相关性最大
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()  # 下载数据集

# ---加载数据
num_train = len(train_x)
num_test = len(test_x)
# ---数据处理/线性归一化
x0_train = np.ones(num_train).reshape(-1)
x_train_5 = (train_x[:, 5] - train_x[:, 5].min(axis=0)) / (train_x[:, 5].max(axis=0) - train_x[:, 5].min(axis=0))
x_train_12 = (train_x[:, 12] - train_x[:, 12].min(axis=0)) / (train_x[:, 12].max(axis=0) - train_x[:, 12].min(axis=0))
x_train = np.stack((x0_train, x_train_5, x_train_12), axis=1)
y_train = tf.where(train_y < 0.5*(train_y.max()+train_y.min()), 0., 1.)

x0_test = np.ones(num_test).reshape(-1)
x_test_5 = (test_x[:, 5] - test_x[:, 5].min(axis=0)) / (test_x[:, 5].max(axis=0) - test_x[:, 5].min(axis=0))
x_test_12 = (test_x[:, 12] - test_x[:, 12].min(axis=0)) / (test_x[:, 12].max(axis=0) - test_x[:, 12].min(axis=0))
x_test = np.stack((x0_test, x_test_5, x_test_12), axis=1)
y_test = test_y

X_train = tf.cast(tf.concat(x_train, axis=1), tf.float32)
X_test = tf.cast(tf.concat(x_test, axis=1), tf.float32)

Y_train = tf.constant(y_train, tf.float32)#.reshape(-1, 1)
Y_test = tf.constant(y_test.reshape(-1, 1), tf.float32)
print(y_train,Y_train)
# ---设置超参数
learn_rate = 0.1
iter = 3000
display_step = 300
# ---设置模型参数初值w
w = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)
# ---训练模型
ce = []  # 每次迭代的交叉熵损失
acc = []  # 准确率

# print(Y_train)
for i in range(0, iter + 1):
    with tf.GradientTape(persistent=True) as tape:
        pred = 1 / (1 + tf.exp(-tf.matmul(X_train, w)))  # sigmoid函数 每个样本的预测概率
        Loss = -tf.reduce_mean(Y_train * tf.math.log(pred) + (1 - Y_train) * tf.math.log(1 - pred))  # 平均交叉熵损失

        #pred_test = tf.matmul(X_test, w)
        #Loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - pred_test))

    dL_dw = tape.gradient(Loss, w)
    w.assign_sub(learn_rate*dL_dw)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred.numpy() < 0.5, 0., 1.), Y_train), tf.float32))
    # print(pred)
    ce.append(Loss)
    acc.append(accuracy)

    if i % display_step == 0:
        print(pred)
        print("i:%i, Acc:%f, Loss:%f" % (i, accuracy, Loss))

from mpl_toolkits.mplot3d import Axes3D

X1,X2 = np.meshgrid(X_train[:,1], X_train[:,2])
Y_PRED = w[1]*X1+w[2]*X2+w[0]
fig = plt.figure(figsize = (8,6))
ax3d = Axes3D(fig)
ax3d.scatter(X_train[:,1], X_train[:,2], Y_train)
ax3d.plot_surface(X1,X2,Y_PRED,cmap="coolwarm")
ax3d.set_xlabel('Room',color='r',fontsize=16)
ax3d.set_ylabel('LowIncom',color='r',fontsize=16)
ax3d.set_zlabel('Price',color='r',fontsize=16)
ax3d.set_yticks([1,2,3])  #y轴坐标轴刻度
# ax3d.view_init(10,-70)  #改变观察视角（水平视角，水平旋转的角度）
plt.show()