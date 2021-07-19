import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

''' boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()  # 下载数据集

# ---加载数据
x_train = train_x[:, 12]
y_train = train_y
x_test = test_x[:, 12]
y_test = test_y '''

x_train = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                      106.69, 138.05, 53.75,  46.91,   68.00, 63.02, 81.26,  86.21])
# house_room = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
y_train = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 
                       62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])
# ---设置超参数
learn_rate = 0.00001  # 学习率
iter = 200  # 迭代次数
display_step = 10  # 输出结果的间隔
# ---设置模型参数初值w0，b0
np.random.seed(612)
w = np.random.randn()
b = np.random.randn()
# ---训练模型
start = time.perf_counter()  # 记录程序起始时间

mse_train = []  # 保存每次迭代的损失值
mse_test = []
for i in range(0, iter + 1):

    ''' with tf.GradientTape(persistent=True) as tape:
        pred_train = w * x_train + b
        loss_train = 0.5 * tf.reduce_mean(tf.square(y_train - pred_train))  # 损失函数

        pred_test = w * x_test + b
        loss_test = 0.5 * tf.reduce_mean(tf.square(y_test - pred_test)) '''
    dL_dw = np.mean(x_train*(w*x_train+b-y_train))
    dL_db = np.mean(w*x_train+b-y_train)

    w = w - learn_rate*dL_dw
    b = b - learn_rate*dL_db

    # pred_test = x_test*w + b
    # Loss_test = (np.mean(np.square(y_test-pred_test)))/2
    pred_train = x_train*w + b
    Loss_train = (np.mean(np.square(y_train-pred_train)))/2
    
    # mse_test.append(Loss_test)
    mse_train.append(Loss_train)

    if i % display_step == 0:
        print("i:%i, loss_train:%f\n" % (i, Loss_train))
end = time.perf_counter()  # 记录程序结束时间
print("程序执行时间：", end - start)
# ---结果可视化
plt.rcParams["font.sans-serif"] = "SimHei"  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

plt.figure(figsize=(10, 10))
# 第一幅--->训练集散点图 和 梯度下降法所求折线图
plt.subplot(131)
plt.scatter(x_train, y_train, color='b', label='sale_data')
plt.plot(x_train, pred_train, color='r', label='predict_model')
plt.legend(loc='upper left')

# 第二幅--->训练集和测试集的损失值
plt.subplot(132)
plt.plot(mse_train)
plt.xlabel('Iteration')
plt.ylabel('loss')
# plt.subplot(222)
# plt.plot(mse_train, color='b', linewidth=3, label='train loss')
# plt.plot(mse_train, color='b', linewidth=1.5, label='test loss')
# plt.xlabel('迭代次数', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.legend(loc='upper right')

# 第三幅--->训练集的房价和梯度下降法所预测的房价
plt.subplot(133)
plt.plot(y_train, color='b', marker='o', label='true_price')
plt.plot(pred_train, color='r', marker='.', label='predict')
plt.legend()
# plt.subplot(223)
# plt.plot(y_train, color='b', marker='o', label='true_price')
# plt.plot(pred_train, color='r', marker='.', label='predict')
# plt.legend()

# 第四幅--->测试集的房价和梯度下降法所预测的房价
# plt.subplot(224)
# plt.plot(y_test, color='b', marker='o', label='true_price')
# plt.plot(pred, color='r', marker='.', label='predict')
# plt.legend()

plt.show()

''' x_train = (train_x[:,12] - train_x[:,12].min(axis=0)) / (train_x[:,12].max(axis=0) - train_x[:,12].min(axis=0))
y_train = (train_y - train_y.min(axis=0)) / (train_y.max(axis=0) - train_y.min(axis=0))

x_test = (test_x[:,12] - test_x[:,12].min(axis=0)) / (test_x[:,12].max(axis=0) - test_x[:,12].min(axis=0))
y_test = test_y

learn_rate = 0.0001  # 学习率
iter = 100  # 迭代次数
display_step = 10  # 输出结果的间隔

np.random.seed(612)
w = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)
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
        print("i:%i, Acc:%f, loss:%f" % (i, accuracy, loss)) '''
