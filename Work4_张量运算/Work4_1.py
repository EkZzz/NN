import numpy
import tensorflow as tf
# 定义数据
x_constant = tf.constant(numpy.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]))
y_constant = tf.constant(numpy.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]))

# 求平均值
x_mean = tf.reduce_mean(x_constant)
y_mean = tf.reduce_mean(y_constant)

# 公式求w和b
w = tf.reduce_sum((x_constant - x_mean)*(y_constant - y_mean))/tf.reduce_sum((x_constant - x_mean)**2)
b = y_mean - w*x_mean

# 打印
print(w)
print(f'w:{w}\nb:{b}')