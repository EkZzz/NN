import numpy

# 定义所需数据列表
x0_list = [1 for _ in range(10)]
# x0_list = numpy.ones()
x1_list = [64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]
x2_list = [2, 3, 4, 2, 3, 4, 2, 4, 1, 3]
y_list = [62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84]

# 堆叠产生X
# # x_list = numpy.stack([x0_list, x1_list, x2_list], axis=1)
# matrix array 结果相同
X_list = numpy.matrix([x0_list, x1_list, x2_list]).transpose()

# reshape转置产生Y
Y_list = numpy.matrix(y_list).reshape((10,1))

# 运算
w_list = (((numpy.linalg.inv(numpy.dot(X_list.T, X_list))).dot(X_list.T)).dot(Y_list))

# 打印结果
print(f'w\'s shape: {w_list.shape}')
print(f'X: \n{X_list}\nY: {Y_list}\nw: {w_list}')


# print((numpy.matrix(x0_list)).shape)
# print(numpy.dot((numpy.matrix(x0_list).T), (numpy.matrix(x0_list))))
# print(f'w\'s shape: {w_list.shape}')
# print(f'X: \n{X_list}\nY: {Y_list}\nw: {w_list}')
# x_mean = numpy.mean(x_list)
# y_mean = numpy.mean(y_list)

# w = sum((x_list - x_mean)*(y_list - y_mean))/sum((x_list - x_mean)**2)
# b = y_mean - w*x_mean
# print(f'w:{w}\nb:{b}')