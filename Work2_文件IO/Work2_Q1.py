import numpy

random_list = [0 for _ in range(1000)]      # 初始化一个大小为1000的0列表

def work2_q1_input_output(random_list=[]):
    # 输入一个整数
    # 打印其整数倍的随机数
    i = int(input("请输入一个1-100之间的整数："))
    if i>0 and i<=100:
        j = 0
        num = 0
        for num in range(0,1000):
            if num % i == 0:
                j += 1
                print('%4d  %4d  %.30f' % (j, num, random_list[num]))

numpy.random.seed(612)      # 设置随机种子
for i in range(0, 1000):    #给列表赋随机值
    random_list[i] = numpy.random.random()

work2_q1_input_output(random_list) #调用函数输入整数和打印满足条件数据




