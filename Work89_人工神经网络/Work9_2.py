import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'

fashion_mnist=tf.keras.datasets.fashion_mnist
(train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()
names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套','凉鞋', '衬衫', '运动鞋','包', '短靴']

#对属性进行归一化,使取值范围在0-1之间，同时转换为tensor张量，标签值转换为张量，0-9之间的整数
X_train,X_test=tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
Y_train,Y_test=tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

# 建立Sequential模型，使用add方法添加层+
model=tf.keras.Sequential()
# model = tf.keras.models.load_model('C:\\Users\\小EK\\Desktop\\PyWork\\Work89_人工神经网络\\modle\\fashion_weight.h5')

try:
    model = tf.keras.models.load_model('C:\\Users\\小EK\\Desktop\\PyWork\\Work89_人工神经网络\\modle\\fashion_weight.h5')
except:
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#Flatten不进行计算，将输入的二维数组转换为一维数组
    model.add(tf.keras.layers.Dense(128,activation="relu"))#添加隐含层，隐含层是全连接层，128个结点,激活函数使用relu函数
    model.add(tf.keras.layers.Dense(10,activation="softmax"))#添加输出层，输出层使全连接层，激活函数是softmax函数

    #配置训练方法
    #优化器使用adam,损失函数使用稀疏交叉熵损失函数，准确率使用稀疏分类准确率函数
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

    #训练模型
    #使用训练集中的数据训练，从中划分百分之20作为测试数据,用在每轮训练后评估模型的性能，每个小批量使用64条数据，训练5轮
    model.fit(X_train,Y_train,batch_size=64,epochs=5,validation_split=0.2)
    model.save("C:\\Users\\小EK\\Desktop\\PyWork\\Work89_人工神经网络\\modle\\fashion_weight.h5", overwrite=True, save_format=None)

#使用测试集评估模型,verbose=2表示为每一轮输出一行记录
model.evaluate(X_test,Y_test,verbose=2)

# #使用模型
for i in range(4):
    num = np.random.randint(1,10000)

    plt.subplot(1,4,i+1)
    plt.axis("off")
    plt.imshow(test_x[num],cmap="gray")
    y_pred=np.argmax(model.predict(test_x[num].reshape(1,28,28)))
    plt.title("原值:"+names[test_y[num]]+"\n预测值:"+names[y_pred])
plt.show()

