import numpy

a = numpy.random.randint(0, 100, size=(4, 4))  # 随机数组范围为0-100
b = numpy.array([])
print('产生的随机4*4数组')
print(a)
for i in range(0, 4):  # 每一列最小
    y = a[0, i]
    for j in range(0, 3):
        if y > a[j + 1, i]:
            y = a[j + 1, i]
    b = numpy.append(b, y)

for i in range(0, 4):  # 每一行最小
    x = a[i, 0]
    for j in range(0, 3):
        if x > a[i, j + 1]:
            x = a[i, j + 1]
    b = numpy.append(b, x)

b = numpy.sort(b)  # 从小到大排序
b.resize(4, 2)

print(b)
