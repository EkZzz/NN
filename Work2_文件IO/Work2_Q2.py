import numpy
# 定义数据
x_list = numpy.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y_list = numpy.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

# 求平均值
x_mean = numpy.mean(x_list)
y_mean = numpy.mean(y_list)

# 公式求sum和b
w = sum((x_list - x_mean)*(y_list - y_mean))/sum((x_list - x_mean)**2)
b = y_mean - w*x_mean

# 打印
print(f'w:{w}\nb:{b}')