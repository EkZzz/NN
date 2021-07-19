import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()  # 下载数据集

# ---加载数据
x_train = train_x[:, 12]
y_train = train_y
x_test = test_x[:, 12]
y_test = test_y
# ---设置超参数
learn_rate = 0.009  # 学习率
iter = 3000  # 迭代次数
display_step = 300  # 输出结果的间隔
# ---设置模型参数初值w0，b0
np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())
# ---训练模型
start = time.perf_counter()  # 记录程序起始时间

mse_train = []  # 保存每次迭代的损失值
mse_test = []
for i in range(0, iter + 1):

    with tf.GradientTape(persistent=True) as tape:
        pred_train = w * x_train + b
        loss_train = 0.5 * tf.reduce_mean(tf.square(y_train - pred_train))  # 损失函数

        pred_test = w * x_test + b
        loss_test = 0.5 * tf.reduce_mean(tf.square(y_test - pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dl_dw = tape.gradient(loss_train, w)  # 求偏导
    dl_db = tape.gradient(loss_train, b)

    w.assign_sub(learn_rate * dl_dw)  # 迭代更新
    b.assign_sub(learn_rate * dl_db)

    del tape
    if i % display_step == 0:
        print("i:%i, loss_train:%f, loss_test:%f" % (i, loss_train, loss_test))
end = time.perf_counter()  # 记录程序结束时间
print("程序执行时间：", end - start)

print(w,b)

# ---结果可视化
plt.rcParams["font.sans-serif"] = "SimHei"  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

plt.figure(figsize=(10, 10))
# 第一幅--->训练集散点图 和 梯度下降法所求折线图
plt.subplot(221)
plt.scatter(x_train, y_train, color='b', label='data')
plt.plot(x_train, pred_train, color='r', label='model')
plt.legend(loc='upper left')    

# 第二幅--->训练集和测试集的损失值
plt.subplot(222)
plt.plot(mse_train, color='b', linewidth=3, label='train loss')
plt.plot(mse_test, color='r', linewidth=1.5, label='test loss')
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right')

# 第三幅--->训练集的房价和梯度下降法所预测的房价
plt.subplot(223)
plt.plot(y_train, color='b', marker='o', label='true_price')
plt.plot(pred_train, color='r', marker='.', label='predict')
plt.legend()

# 第四幅--->测试集的房价和梯度下降法所预测的房价
plt.subplot(224)
plt.plot(y_test, color='b', marker='o', label='true_price')
plt.plot(pred_test, color='r', marker='.', label='predict')
plt.legend()

plt.show()

'''learn_rate = 0.1  # 学习率
iter = 30  # 迭代次数
display_step = 2  # 输出结果的间隔

ce = []  # 每次迭代的交叉熵损失
acc = []  # 准确率
for i in range(0, iter + 1):

    with tf.GradientTape(persistent=True) as tape:
        pred = 1 / (1 + tf.exp(-(w * x_train + b)))  # sigmoid函数 每个样本的预测概率
        loss = -tf.reduce_mean(y_train * tf.math.log(pred) + (1 - y_train) * tf.math.log(1 - pred))  # 平均交叉熵损失

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(pred.numpy() < 0.5, 0., 1.), y_train), tf.float32))

    ce.append(loss)
    acc.append(accuracy)

    dl_dw = tape.gradient(loss, w)  # 求偏导
    dl_db = tape.gradient(loss, b)

    w.assign_sub(learn_rate * dl_dw)  # 迭代更新
    b.assign_sub(learn_rate * dl_db)

    if i % display_step == 0:
        print("i:%i, Acc:%f, loss:%f" % (i, accuracy, loss))
'''