import numpy as np
import tensorflow as tf

house_area = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                      106.69, 138.05, 53.75,  46.91,   68.00, 63.02, 81.26,  86.21])
# house_room = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
house_price = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 
                       62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

x_mean = tf.reduce_mean(house_area)
y_mean = tf.reduce_mean(house_price)

# 公式求w和b
w = tf.reduce_sum((house_area - x_mean)*(house_price - y_mean))/tf.reduce_sum((house_area - x_mean)**2)
b = y_mean - w*x_mean

# 面积输入
def area_input_detector():
    chance = 3
    while chance:
        chance -= 1
        try:
            data = input('请输入面积（一个20-500之间的实数）：')
            if data == '' or data.isspace():
                print('对不起，你的输入为空！')
                print('您还有%d次机会，请重新输入。' % chance)
                print('-'*30)
                continue
            data = float(data)
        except ValueError as ve:
            print('对不起，你的输入类型有误，并非实数数据！')
        else:
            if 20 <= data <= 500:
                return data
            else:
                print('对不起，您输入的数字范围不正确!')
        print('您还有%d次机会，请重新输入。' % chance)
        print('-'*30)
    print('对不起，您已经3次输入错误，程序退出。')

area_input_data = area_input_detector()
print(f'面积：{area_input_data}\n房价：{float(w*area_input_data+b)}')