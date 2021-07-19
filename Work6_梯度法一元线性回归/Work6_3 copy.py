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
y_train = train_y

x0_test = np.ones(num_test).reshape(-1)
x_test_5 = (test_x[:, 5] - test_x[:, 5].min(axis=0)) / (test_x[:, 5].max(axis=0) - test_x[:, 5].min(axis=0))
x_test_12 = (test_x[:, 12] - test_x[:, 12].min(axis=0)) / (test_x[:, 12].max(axis=0) - test_x[:, 12].min(axis=0))
x_test = np.stack((x0_test, x_test_5, x_test_12), axis=1)
y_test = test_y

X_train = tf.cast(tf.concat(x_train, axis=1), tf.float32)
X_test = tf.cast(tf.concat(x_test, axis=1), tf.float32)
Y_train = tf.constant(y_train.reshape(-1, 1), tf.float32)
Y_test = tf.constant(y_test.reshape(-1, 1), tf.float32)

print(X_train.shape)
# ---设置超参数
learn_rate = 0.009
iter = 2500
display_step = 250
# ---设置模型参数初值w
w = tf.Variable(np.random.randn(3, 1),dtype=tf.float32)
# ---训练模型
mse_train = []  # 保存每次迭代的损失值
mse_test = []

for i in range(0, iter + 1):

    with tf.GradientTape(persistent=True) as tape:
        pred_train = tf.matmul(X_train,w)
        loss_train = 0.5 * tf.reduce_mean(tf.square(Y_train - pred_train))  # 损失函数

        pred_test = tf.matmul(X_test,w)
        loss_test = 0.5 * tf.reduce_mean(tf.square(Y_test - pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dl_dw = tape.gradient(loss_train, w)  # 求偏导

    w.assign_sub(learn_rate * dl_dw)  # 迭代更新

    if i % display_step == 0:
        print("i:%i, loss_train:%f, loss_test:%f" % (i, loss_train, loss_test))
# ---结果可视化
plt.rcParams["font.sans-serif"] = "SimHei"  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

fig = plt.figure(figsize=(18, 6))

# 第1幅--->训练集和测试集的损失值
plt.subplot(131)
plt.plot(mse_train, color='b', linewidth=3, label='train loss')
plt.plot(mse_test, color='b', linewidth=1.5, label='test loss')
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right')

# 第2幅--->训练集的房价和梯度下降法所预测的房价
plt.subplot(132)
plt.plot(y_train, color='b', marker='o', label='true_price')
plt.plot(pred_train, color='r', marker='.', label='predict')
plt.legend()
plt.ylabel('Price', fontsize=14)

# 第3幅--->测试集的房价和梯度下降法所预测的房价
plt.subplot(133)
plt.plot(y_test, color='b', marker='o', label='true_price')
plt.plot(pred_test, color='r', marker='.', label='predict')
plt.legend()
plt.ylabel('Price', fontsize=14)

plt.show()

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