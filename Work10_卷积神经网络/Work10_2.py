import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import time
from tensorflow.keras import layers,Sequential

plt.rcParams['font.sans-serif']=['SimHei']

cifar10=tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train,x_test=tf.cast(x_train,dtype=tf.float32)/255.0,tf.cast(x_test,dtype=tf.float32)/255.0
y_train,y_test=tf.cast(y_train,dtype=tf.int32),tf.cast(y_test,tf.int32)

model=Sequential([
    #unit1
    layers.Conv2D(16,kernel_size=(3,3),padding="same",activation=tf.nn.relu,input_shape=x_train.shape[1:]),
    layers.Conv2D(16,kernel_size=(3,3),padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2)),

    #unit2
    layers.Conv2D(32,kernel_size=(3,3),padding="same",activation=tf.nn.relu),
    layers.Conv2D(32,kernel_size=(3,3),padding="same",activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2)),
    
    #unit3
    layers.Flatten(),
    layers.Dense(1024,activation="relu"),
    layers.Dense(10,activation="softmax")
])
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

start = time.perf_counter()

model.fit(x_train, y_train,batch_size=64,epochs=5,validation_split=0.2)

end = time.perf_counter()
print("模型训练时间: ",end-start)
model.save_weights("C:\\Users\\小EK\\Desktop\\PyWork\\Work10_卷积神经网络\\model\\cifar_weight.h5", overwrite=True, save_format=None)


print("模型评估结果为：")
model.evaluate(x_test,y_test,verbose=2)