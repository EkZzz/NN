''' 手写数据集 '''
import matplotlib.pyplot as mpl
import numpy as np
import tensorflow as tf
import PIL.Image as im

mpl.rcParams["font.sans-serif"] = ['SimHei']
mpl.rcParams["axes.unicode_minus"] = False

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

mpl.figure()

for i in range(16):
    index = np.random.randint(1,10000)
    mpl.subplot(4,4,i+1)
    mpl.axis('off')
    if np.random.randint(0, 2) % 2:
        mpl.imshow(im.fromarray(test_x[index]).transpose(im.ROTATE_270), cmap='gray')
    else:
        mpl.imshow(im.fromarray(test_x[index]).crop((0,0,20,20)), cmap='gray')

    mpl.title('标签值：'+str(test_y[index]), fontsize=14)

mpl.suptitle('MNIST测试集样本', fontdict={'color':'r'}, fontsize=20)
mpl.show()