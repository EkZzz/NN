import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


plt.rcParams['font.sans-serif']='SimHei'
#导入mnist的训练集和测试集
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()
img_testx =test_x
img_testy =test_y
#对属性进行归一化,使取值范围在0-1之间，同时转换为tensor张量，标签值转换为张量，0-9之间的整数
X_train,X_test=tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
Y_train,Y_test=tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)
X_img=X_test
#建立Sequential模型，使用add方法添加层+
model=tf.keras.Sequential()
try:
    model = tf.keras.models.load_model("C:\\Users\\小EK\\Desktop\\PyWork\\Work89_人工神经网络\\modle\\mnist_weight.h5")
except:
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#Flatten不进行计算，将输入的二维数组转换为一维数组
    model.add(tf.keras.layers.Dense(128,activation="relu"))#添加隐含层，隐含层是全连接层，128个结点,激活函数使用relu函数
    model.add(tf.keras.layers.Dense(10,activation="softmax"))#添加输出层，输出层使全连接层，激活函数是softmax函数

    #配置训练方法
    #优化器使用adam,损失函数使用稀疏交叉熵损失函数，准确率使用稀疏分类准确率函数
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

    # print(X_train.shape)
    # print(Y_train.shape)

    # #训练模型
    # #使用训练集中的数据训练，从中划分百分之20作为测试数据,用在每轮训练后评估模型的性能，每个小批量使用64条数据，训练5轮
    model.fit(X_train,Y_train,batch_size=64,epochs=5,validation_split=0.2)

    # #使用测试集评估模型
    
    model.save("C:\\Users\\小EK\\Desktop\\PyWork\\Work89_人工神经网络\\modle\\mnist_weight.h5", overwrite=True, save_format=None)
model.evaluate(X_test,Y_test,verbose=2)
#使用模型预测随机四个数据
for i in range(5):
    num = np.random.randint(1,10000)
    plt.subplot(1,5,i+1)
    plt.axis("off")
    plt.imshow(test_x[num],cmap="gray")
    #argmax取出值最大的索引,predict中参数的数据范围和维数与训练集一致
    # y_pred = np.argmax(model.predict([[X_test[num]]]))
    # tensor = tf.convert_to_tensor(X_test[num].numpy().reshape(1,28*28))
    y_pred=np.argmax(model.predict(tf.convert_to_tensor(X_test[num].numpy().reshape(1,28,28))))
    plt.title("原值="+str(test_y[num])+"\n预测值："+str(y_pred))
plt.show()

model.load_weights("C:\\Users\\小EK\\Desktop\\PyWork\\Work89_人工神经网络\\modle\\mnist_weight.h5")
#使用模型预测自己的手写数据集
img_arr= []
lbl_arr= []
for i in range(0,10):
    img = Image.open(f"C:\\Users\\小EK\\Desktop\\PyWork\\MT\\number_data\\result{i}.png")
    img_array=np.array(img)
    img_arr.append(img_array)
    lbl_arr.append(i)
# img_arr = np.array(img_arr)
# lbl_arr = np.array(lbl_arr)
img_test=tf.cast(img_arr,tf.float32)
lbl_test=tf.cast(lbl_arr,tf.int16)

for i in range(10):
    # num = np.random.randint(1,10)
    plt.subplot(1,10,i+1)
    plt.axis("off")
    plt.imshow(img_arr[i],cmap="gray")
    #argmax取出值最大的索引,predict中参数的数据范围和维数与训练集一致
    # y_pred = np.argmax(model.predict([[X_test[num]]]))
    # tensor = tf.convert_to_tensor(X_test[num].numpy().reshape(1,28*28))
    y_pred=np.argmax(model.predict(tf.convert_to_tensor(img_test[i].numpy().reshape(1,28,28))))
    plt.title("原值="+str(lbl_arr[i])+"\n预测值："+str(y_pred))
plt.show()