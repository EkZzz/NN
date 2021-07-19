''' 1. 打印1-1000中，所有能被20整除的数，并打印序号。序号从1开始。 '''
def Q1():
    i = 0
    num = 0
    while (num := num + 20) <= 1000:
        i += 1
        print('%4d  %4d' % (i, num))


''' 2.用户输入一个1-100之间的数字。打印1-1000中，所有可以被这个数字整除的数字，并打印
序号。序号从1开始，依次加1. '''
def Q2():
    i = int(input("请输入一个1-100之间的整数："))
    j = 0
    num = 0
    while (num := num + i) <= 1000:
        j += 1
        print('%4d  %4d' % (j, num))


''' 3.输出提示信息：“请输入1-100之间的整数：”接收用户键盘输入，如果输入的是1-100之间
的整数，输出“您输入的是整数：xx”，程序结束运行；如果输入的不是整数，或不在指定的范围，
输出“对不起，您的输入无效，请重新输入！”，直到用户输入正确为止。
'''
def Q3():
    chance = 3
    while chance:
        chance -= 1
        try:
            data = input('请输入一个1-100之间的整数：')
            if data == '' or data.isspace():
                print('对不起，你的输入为空！')
                print('您还有%d次机会，请重新输入。' % chance)
                print('-'*30)
                continue
            data = int(data)
        except ValueError as ve:
            print('对不起，你的输入类型有误，并非整型数据！')
        else:
            if 1 <= data <= 100:
                print('您输入的是整数：%d' % data)
                return
            else:
                print('对不起，您输入的数字范围不正确!')
        print('您还有%d次机会，请重新输入。' % chance)
        print('-'*30)
    print('对不起，您已经3次输入错误，程序退出。')


''' 4.用户输入一个1-100之间的整数。在屏幕上输出1-1000中，所有可以被这个输入数字整除
的整数，并把它们写入文本文件中。 '''

# 接收用户输入，并判断是否为1-100之间的整数
def Q4_1():
    chance = 3
    while chance:
        chance -= 1
        try:
            num = int(input('请输入一个1-100之间的整数：'))
        except ValueError as ve:
            print('对不起，你的输入类型有误，并非整型数据！')
        else:
            if 1 <= num <= 100:
                print('您输入的是整数：%d' % num)
                return num
            else:
                print('对不起，您输入的数字范围不正确!')
        print('您还有%d次机会，请重新输入。' % chance)
        print('-'*30)
    print('对不起，您已经3次输入错误，程序退出。')

# 根据用户输入，在屏幕上输出1-1000中，所有可以被这个数字整除的数字，并打印序号
def Q4_2(data):
    i = data
    j = 0
    num = 0
    res = ''
    while (num := num + i) <= 1000:
        j += 1
        res += '%4d  %4d\n' % (j, num)
    print(res)
    return res

def Q4():
    dataIn = Q4_1()                 # 输入整数
    dataOut = Q4_2(dataIn)          # 输出小于1000所有可以被整除的数

    with open('C:\\Users\\小EK\\Desktop\\Work1\\%d的倍数.txt' % dataIn, 'a') as f:
        f.write(dataOut)
        print('数据写入成功！')
    f.close()


Q4()
