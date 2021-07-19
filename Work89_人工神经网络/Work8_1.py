import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PrtalWIdth', 'Species']
tf_iris_train = pd.read_csv(train_path, names=COLUMN_NAMES, header=0)
tf_iris_test = pd.read_csv(test_path, names=COLUMN_NAMES, header=0)
iris_train = np.array(tf_iris_train)  # 下载数据集
iris_test = np.array(tf_iris_test) # 下载测试集

# ---加载数据
x_train = iris_train[:,0:4]
y_train = iris_train[:,4]
x_test = iris_test[:,0:4]
y_test = iris_test[:,4]
num_train= len(y_train)   # 属性长度
num_test = len(y_test)

# 属性中心化
x_train = x_train - tf.reduce_mean(x_train, axis=0)
# y_train = y_train.reshape(-1,1)
x_test = x_test - tf.reduce_mean(x_test, axis=0)
# y_test = y_test.reshape(-1,1)

# ---数据处理/线性归一化
x0_train = np.ones(num_train).reshape(-1,1)
x0_test = np.ones(num_test).reshape(-1,1)

X_train = tf.cast(tf.concat((x0_train,x_train), axis=1), tf.float32)
Y_train = tf.one_hot(tf.constant(y_train, tf.int32),3)
X_test = tf.cast(tf.concat((x0_test,x_test), axis=1), tf.float32)
Y_test = tf.one_hot(tf.constant(y_test, tf.int32),3)

# ---设置超参数
learn_rate = 0.6
iter = 60
display_step = 10
# ---设置模型参数初值w
np.random.seed(612)
w = tf.Variable(np.random.randn(5, 3), dtype=tf.float32)
# ---训练模型
ce_train = []  # 每次迭代的交叉熵损失
acc_train = []  # 准确率
ce_test = []  # 每次迭代的交叉熵损失
acc_test = []  # 准确率


for i in range(0, iter + 1):
    with tf.GradientTape(persistent=True) as tape:
        pred_train = tf.nn.softmax(tf.matmul(X_train,w))
        loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train, y_pred=pred_train))

    pred_test = tf.nn.softmax(tf.matmul(X_test,w))
    loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=pred_test))

    grads = tape.gradient(loss_train, w)
    w.assign_sub(learn_rate*grads)

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_train.numpy(), axis=1), y_train), tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_test.numpy(), axis=1), y_test), tf.float32))

    ce_train.append(loss_train)
    acc_train.append(accuracy_train)
    ce_test.append(loss_test)
    acc_test.append(accuracy_test)

    if i % display_step == 0:
        print("i:%i, Acc_tra:%f, Loss_tra:%f\tAcc_te:%f, Loss_te:%f" % (i, accuracy_train, loss_train, accuracy_test, loss_test))

plt.figure(figsize=(5,3))
plt.subplot(121)
plt.plot(ce_train, c='b', label='Losstrain')
plt.plot(acc_train, c='r', label='acctrain')
plt.ylim((0, 1))
plt.legend()
plt.subplot(122)
plt.plot(ce_test, c='g', label='Losstest')
plt.plot(acc_test, c='r', label='acctest')
plt.ylim((0, 1))
plt.legend()

plt.show()

