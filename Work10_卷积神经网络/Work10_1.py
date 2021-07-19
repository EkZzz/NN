import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

mnist=tf.keras.datasets.mnist

(train_x,train_y),(test_x,test_y)=mnist.load_data()

X_train,X_test=tf.cast(train_x,dtype=tf.float32)/255.0,tf.cast(test_x,dtype=tf.float32)/255.0
y_train,y_test=tf.cast(train_y,dtype=tf.int32),tf.cast(test_y,dtype=tf.int32)

X_train=train_x.reshape(60000,28,28,1)
X_test=test_x.reshape(10000,28,28,1)

model=tf.keras.Sequential([
    #unit1
    tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    #unit2
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    #unit3
    tf.keras.layers.Flatten(),
    #unit4
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')

])

#model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

start = time.perf_counter()

model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

end = time.perf_counter()
print("模型训练时间: ",end-start)

print("模型评估结果为：")
model.evaluate(X_test, y_test, verbose=2)