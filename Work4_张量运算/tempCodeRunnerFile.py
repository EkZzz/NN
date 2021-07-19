x_mean = tf.reduce_mean(x_constant)
y_mean = tf.reduce_mean(y_constant)

# 公式求w和b
w = tf.reduce_sum((x_constant - x_mean)*(y_constant - y_mean))/tf.reduce_sum((x_constant - x_mean)**2)
b = y_mean - w*x_mean