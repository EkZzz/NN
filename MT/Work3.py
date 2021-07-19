import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover','Dress', 
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 训练集长度及形状
print(train_x.shape)

# 测试集长度
print(len(test_x))

# 标签
plt.imshow(Image.fromarray(train_x[1]))
plt.title(class_names[train_y[1]])
plt.axis('off')
plt.show()

train_x_aug1 = []
for i in range(0,10):
    img = Image.fromarray(train_x[i])
#     原图
    train_x_aug1.append(np.array(img))
#     转置
    train_x_aug1.append(np.array(img.transpose(Image.TRANSPOSE)))
#     上下翻转
    train_x_aug1.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
#     顺时针10
    train_x_aug1.append(np.array(img.rotate(-10)))
#     逆时针10
    train_x_aug1.append(np.array(img.rotate(10)))
#     水平镜像
    train_x_aug1.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))

# len(train_x_aug1)

change_mode = ['原图','转置','上下翻转','顺时针10','逆时针10','水平镜像']

fig = plt.figure()
plt.suptitle('Fashion Mnist数据增强', color='r', fontsize=20)
for i in range(0, 60):
        plt.subplot(10,6,i+1)
        plt.axis('off')
        plt.imshow(Image.fromarray(train_x_aug1[i]))
        if i<6:
            plt.title(change_mode[i])
plt.show()

# (5)

train_x_aug2 = []
for i in range(0,10):
    img = Image.fromarray(train_x[i])
#     原图
    train_x_aug2.append(np.array(img))
#     转置
    train_x_aug2.append(np.array(img.transpose(Image.TRANSPOSE)))
#     上下翻转
    train_x_aug2.append(np.array(img.transpose(Image.FLIP_TOP_BOTTOM)))
#     顺时针
    angle1 = np.random.randint(-360,0)
    train_x_aug2.append(np.array(img.rotate(angle1)))
#     逆时针
    angle2 = np.random.randint(0,360)
    train_x_aug2.append(np.array(img.rotate(angle2)))
#     水平镜像
    train_x_aug2.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))

# len(train_x_aug2)

change_mode2 = ['原图','转置','上下翻转','顺时针','逆时针','水平镜像']

fig = plt.figure()
plt.suptitle('Fashion Mnist数据增强', color='r', fontsize=20)
for i in range(0, 60):
        plt.subplot(10,6,i+1)
        plt.axis('off')
        plt.imshow(Image.fromarray(train_x_aug2[i]))
        if(i<6):
            plt.title(change_mode2[i])
plt.show()

# (6)

import random

train_x_aug3 = []
for i in range(0,100):
    img = Image.fromarray(train_x[i])
    for j in range(0,5):
        choice = [np.array(img), np.array(img.transpose(Image.TRANSPOSE)), np.array(img.transpose(Image.FLIP_TOP_BOTTOM)), 
                 np.array(img.rotate(np.random.randint(-360,0))), np.array(img.rotate(np.random.randint(0,360))), np.array(img.transpose(Image.FLIP_LEFT_RIGHT))]
    
        train_x_aug3.append(random.choice(choice))
# len(train_x_aug3)

for i in range(10):
    index = np.random.randint(0,500)
    plt.subplot(2, 5, i+1)
    plt.imshow(Image.fromarray(train_x_aug3[index]))
    plt.axis('off')

# 扩展

img_1 = Image.fromarray(train_x[0])
# 原图
plt.subplot(221)
plt.imshow(img_1)
plt.axis('off')
(width1, hight1) = img_1.size

# 缩小10%
img_2 = img_1.resize((int(0.9*width1), int(0.9*hight1)))
plt.subplot(222)
plt.axis('off')
plt.imshow(img_2)
(width2, hight2) = img_2.size

# 放大10%
img_3 = img_2.resize((int(1.1*width2), int(1.1*hight2)))
plt.subplot(223)
plt.axis('off')
plt.imshow(img_3)
# print(img_3.size）

# 填充为28*28
img_4 = np.pad(np.array(img_3), ((1,0),(1,0)), 'edge')
plt.subplot(224)
plt.axis('off')
plt.imshow(Image.fromarray(img_4))
print(img_4.shape)