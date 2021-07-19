import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'
#列出所有GPU放在gpus中，设置为内存增长模式
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0],True)

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

TRAIN_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
#读取csv文件，结果是一个pandas二维数据表
df_iris_train = pd.read_csv(train_path,header=0)
df_iris_test = pd.read_csv(test_path,header=0)

#将二维数据表转换为numpy数组
iris_train=np.array(df_iris_train)#(120,5)
iris_test=np.array(df_iris_test)#(30,5)

#取出所有鸢尾花的所有属性值和标签，分开
x_train = iris_train[:,0:4]
y_train = iris_train[:,4]

x_test=iris_test[:,0:4]
y_test=iris_test[:,4]

#对 训练集 和 测试集 属性按列中心化,标准化处理，使均值为0
x_train = x_train-np.mean(x_train,axis=0)
x_test = x_test-np.mean(x_test,axis=0)

#数据集中属性值和标签值都为64浮点数，属性转换为tensor张量和32位浮点数,标签转换为深度为3的独热编码
X_train = tf.cast(x_train,tf.float32)#(120,4)
Y_train = tf.one_hot(tf.constant(y_train,dtype=tf.int32),3)#(120,3)

X_test = tf.cast(x_test,tf.float32)#(30,4)
Y_test = tf.one_hot(tf.constant(y_test,dtype=tf.int32),3)#(30,3)

#设置超参数和显示间隔
learn_rate = 0.5
iter = 50
display_step = 10

#设置模型参数初始值
np.random.seed(612)
W1 = tf.Variable(np.random.randn(4,16),dtype=tf.float32)
B1 = tf.Variable(np.zeros([16]),dtype=tf.float32)

W2 = tf.Variable(np.random.randn(16,3),dtype=tf.float32)
B2 = tf.Variable(np.zeros([3]),dtype=tf.float32)

#训练模型
acc_train=[]
acc_test=[]
cce_train=[]
cce_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #定义网络结构
        #训练集的输出和交叉熵损失
        Hidden_train=tf.nn.relu(tf.matmul(X_train,W1)+B1)#隐含层的输出使用relu函数作为激活函数
        pred_train = tf.nn.softmax(tf.matmul(Hidden_train,W2)+B2)#输出层的输出使用softmax函数作为激活函数
        loss_train=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train,y_pred=pred_train))#使用交叉熵损失函数计算损失
        #测试集的输出和交叉熵损失
        Hidden_test=tf.nn.relu(tf.matmul(X_test,W1)+B1)#隐含层的输出使用relu函数作为激活函数
        pred_test = tf.nn.softmax(tf.matmul(Hidden_test,W2)+B2)#输出层的输出使用softmax函数作为激活函数
        loss_test=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test,y_pred=pred_test))

    #训练集和测试集上的准确率
    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_train.numpy(),axis=1),y_train),tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_test.numpy(),axis=1),y_test),tf.float32))

    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(loss_train)
    cce_test.append(loss_test)

    #更新模型参数
    grads = tape.gradient(loss_train,[W1,B1,W2,B2])
    W1.assign_sub(learn_rate*grads[0])
    B1.assign_sub(learn_rate*grads[1])
    W2.assign_sub(learn_rate*grads[2])
    B2.assign_sub(learn_rate*grads[3])

    if i % display_step == 0:
        print("i: %i, TrainAcc: %f, TrainAcc: %f, TestLoss: %f" % (i, accuracy_train, accuracy_test,loss_test))

plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(cce_train,color="blue",label="train")
plt.plot(cce_test,color="red",label="test")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color="blue",label="train")
plt.plot(acc_test,color="red",label="test")
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.legend()

plt.tight_layout()
plt.show()

