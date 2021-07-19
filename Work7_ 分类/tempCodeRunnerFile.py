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
train_x = iris[:,0:2]
train_y = iris[:,4]
test_x = iris_test[:,0:2]
test_y = iris_test[:,4]

x_train = train_x[train_y != 1]
y_train = train_y[train_y != 1]
x_test = test_x[test_y != 1]
y_test = test_y[test_y != 1]

train_num = len(x_train)   # 属性长度
test_num = len(x_test)

# 属性中心化
x_train = x_train - tf.reduce_mean(x_train, axis=0)
y_train = y_train.reshape(-1,1)
x_test = x_test - tf.reduce_mean(x_test, axis=0)
y_test = y_test.reshape(-1,1)

# ---数据处理/线性归一化
x0_train = np.ones(train_num).reshape(-1,1)
x0_test = np.ones(test_num).reshape(-1,1)

X_train = tf.cast(tf.concat((x0_train,x_train), axis=1), tf.float32)#.reshape(-1, 1)
Y_train = tf.cast(tf.where(y_train==2,1,0), tf.float32)

X_test = tf.cast(tf.concat((x0_test,x_test), axis=1), tf.float32)#.reshape(-1, 1)
Y_test = tf.cast(tf.where(y_test==2,1,0), tf.float32)

# ---设置超参数
learn_rate = 0.5
iter = 50
display_step = 10
# ---设置模型参数初值w
np.random.seed(612)
w = tf.Variable(np.random.randn(3, 1), dtype=tf.float32)
# ---训练模型
ce_train = []  # 每次迭代的交叉熵损失
acc_train = []  # 准确率
ce_test = []  
acc_test = [] 

start = time.perf_counter()
# print(Y_train)
for i in range(0, iter + 1):
    with tf.GradientTape(persistent=True) as tape:
        pred_train = 1 / (1 + tf.exp(-tf.matmul(X_train, w)))  # sigmoid函数 每个样本的预测概率
        Loss_train = -tf.reduce_mean(Y_train * tf.math.log(pred_train) + (1 - Y_train) * tf.math.log(1 - pred_train))  # 平均交叉熵损失
        pred_test = 1 / (1 + tf.exp(-tf.matmul(X_test, w)))  # sigmoid函数 每个样本的预测概率
        Loss_test = -tf.reduce_mean(Y_test * tf.math.log(pred_test) + (1 - Y_test) * tf.math.log(1 - pred_test))  # 平均交叉熵损失



    dL_dw = tape.gradient(Loss_train, w)
    w.assign_sub(learn_rate*dL_dw)

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train.numpy() < 0.5, 0., 1.), Y_train), tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_test.numpy() < 0.5, 0., 1.), Y_test), tf.float32))

    ce_train.append(Loss_train)
    acc_train.append(accuracy_train)
    ce_test.append(Loss_train)
    acc_test.append(accuracy_test)

    if i % display_step == 0:
        print("i:%i, Acc_train:%f, Loss_train:%f, Acc_test:%f, Loss_test:%f" % (i, accuracy_train, Loss_train, accuracy_test, Loss_test))
end = time.perf_counter()  # 记录程序结束时间
print("程序执行时间：", end - start)

plt.figure(figsize=(5,3))
plt.subplot(121)
plt.plot(ce_train, c='b', label='Loss_train')
plt.plot(acc_train, c='r', label='acc_train')
plt.legend()

plt.subplot(122)
plt.plot(ce_test, c='b', label='Loss_test')
plt.plot(acc_test, c='r', label='acc_test')
plt.legend()

plt.show()

plt.subplot(121)
cm_pt = mpl.colors.ListedColormap(["blue", "red"])
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cm_pt)
x_ = [-1.5, 1.5]
y_ = -(w[1]*x_ + w[0])/w[2]
plt.plot(x_, y_, c='g')

plt.subplot(122)
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cm_pt)
x_ = [-1.5, 1.5]
y_ = -(w[1]*x_ + w[0])/w[2]
plt.plot(x_, y_, c='g')

plt.show()