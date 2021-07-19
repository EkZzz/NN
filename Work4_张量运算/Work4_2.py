import numpy
import tensorflow as tf
# 定义数据: 张量类型
x_constant = tf.constant(numpy.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]))
y_constant = tf.constant(numpy.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]))

# 公式求w和b

n = tf.size(x_constant, out_type=tf.float64)
x_sum = tf.reduce_sum(x_constant)
y_sum = tf.reduce_sum(y_constant)

w = (n*tf.reduce_sum(x_constant*y_constant) - x_sum*y_sum) \
    / (n*tf.reduce_sum(x_constant**2) - (x_sum**2))

""" 
Error: cannot compute Mul as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:Mul]
        乘法左右两端Tensor数据的dtype不同
Solution：使用tf.cast(constant, dtype={所要修改的类型（tf.int32 等})
 """

b = (y_sum - w*x_sum )/ n

""" 分子分母加括号！ """

print(y_sum, x_sum)
print (f'w:{w}\nb:{b}')