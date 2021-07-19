import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# config = tf.ConfigProto() 
# config.gpu_options.allow_growth = True
# Check failed: cusolverDnCreate(&cusolver_dn_handle) == CUSOLVER_STATUS_SUCCESS Failed to create cuSolverDN instance.

house_area = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                      106.69, 138.05, 53.75,  46.91,   68.00, 63.02, 81.26,  86.21])
house_room = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
house_price = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 
                       62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

X = tf.stack((np.ones(len(house_area)), house_area, house_room),axis=1)
Y = tf.constant(house_price.reshape(-1,1))

W = tf.matmul(tf.matmul(tf.linalg.inv(np.matmul(tf.transpose(X), X)), tf.transpose(X)), Y)

def area_input_detector():
    chance = 3
    while chance:
        chance -= 1
        try:
            data = input('请输入面积(一个20-500之间的实数)：')
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
    return 0

def room_input_detector():
    chance = 3
    while chance:
        chance -= 1
        try:
            data = input('请输入房间数(一个1-10之间的整数)：')
            if data == '' or data.isspace():
                print('对不起，你的输入为空！')
                print('您还有%d次机会，请重新输入。' % chance)
                print('-'*30)
                continue
            data = int(data)
        except ValueError as ve:
            print('对不起，你的输入类型有误，并非实数数据！')
        else:
            if 1 <= data <= 10:
                return data
            else:
                print('对不起，您输入的数字范围不正确!')
        print('您还有%d次机会，请重新输入。' % chance)
        print('-'*30)
    print('对不起，您已经3次输入错误，程序退出。')
    return 0
area_input_data = area_input_detector()
if area_input_data != 0:
    room_input_data = room_input_detector()
    print(f'面积：{area_input_data}\n房间数：{room_input_data}\n房价：{float(W[1]*area_input_data+W[2]*room_input_data+W[0])}')

    fig =plt.figure(figsize=(8,6))
    ax3d=Axes3D(fig)
    ax3d.scatter(house_area, house_room, house_price, c='b',marker="*")
    ax3d.scatter(area_input_data, room_input_data, float(W[1]*area_input_data+W[2]*room_input_data+W[0]), c='r',marker=".")
    ax3d.set_xlabel("Area", color="r", fontsize=16)
    ax3d.set_ylabel("Room", color="r", fontsize=16)
    ax3d.set_zlabel("Price", color="r", fontsize=16)
    ax3d.set_yticks([1,2,3])
    ax3d.set_zlim3d(30,160)

    plt.show()

    X1,X2 = np.meshgrid(house_area, house_room)
    Y_PRED = W[1]*X1+W[2]*X2+W[0]
    fig = plt.figure(figsize = (8,6))
    ax3d = Axes3D(fig)
    ax3d.plot_surface(X1,X2,Y_PRED,cmap="coolwarm")
    ax3d.set_xlabel('Area',color='r',fontsize=16)
    ax3d.set_ylabel('Room',color='r',fontsize=16)
    ax3d.set_zlabel('Price',color='r',fontsize=16)
    ax3d.set_yticks([1,2,3])  #y轴坐标轴刻度
    # ax3d.view_init(10,-70)  #改变观察视角（水平视角，水平旋转的角度）
    plt.show()