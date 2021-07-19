import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)

TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

df_iris_train = pd.read_csv(train_path, header = 0)
df_iris_test = pd.read_csv(test_path, header = 0)

iris_train = np.array(df_iris_train)
iris_test = np.array(df_iris_test)

x_train = iris_train[:,0:4]
x_test = iris_test[:,0:4]

y_train = iris_train[:,4]
y_test = iris_test[:,4]

x_train = x_train - np.mean(x_train, axis = 0)
x_test = x_test - np.mean(x_test, axis = 0)

X_train = tf.cast(x_train, tf.float32)
X_test = tf.cast(x_test, tf.float32)

Y_train = tf.one_hot(tf.constant(y_train, dtype = tf.int32), 3)
Y_test = tf.one_hot(tf.constant(y_test, dtype = tf.int32), 3)

learn_rate = 0.5
iter = 50
display_step = 10

np.random.seed(612)
W = tf.Variable(np.random.randn(4,3),dtype=tf.float32)
B = tf.Variable(np.zeros([3]), dtype=tf.float32)

acc_train = []
acc_test = []
cce_train = []
cce_test = []

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        PRED_train = tf.nn.softmax(tf.matmul(X_train, W) + B)
        Loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train, y_pred=PRED_train))

    PRED_test = tf.nn.softmax(tf.matmul(X_test,W) + B)
    Loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=PRED_test))

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(), axis=1),y_train),tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(), axis=1), y_test), tf.float32))

    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(Loss_train)
    cce_test.append(Loss_test)

    grads = tape.gradient(Loss_train, [W,B])
    W.assign_sub(learn_rate*grads[0])
    B.assign_sub(learn_rate*grads[1])

    if i % display_step == 0:
        print("i:%i, TrainAcc:%f, TrainLoss:%f, TestAcc:%f, TestLoss:%f" % (i, accuracy_train, Loss_train, accuracy_test, Loss_test))

plt.figure(figsize=(5,3))
plt.subplot(121)
plt.plot(cce_train, c='b', label='Losstrain')
plt.plot(acc_train, c='r', label='acctrain')
plt.ylim((0, 1))
plt.legend()
plt.subplot(122)
plt.plot(cce_test, c='g', label='Losstest')
plt.plot(acc_test, c='r', label='acctest')
plt.ylim((0, 1))
plt.legend()

plt.show()