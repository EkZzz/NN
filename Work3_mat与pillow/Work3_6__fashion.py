''' Fashion MINIS '''

import matplotlib.pyplot as mpl
import numpy as np
import tensorflow as tf
from PIL import Image

mpl.rcParams["font.sans-serif"] = ['SimHei']
mpl.rcParams["axes.unicode_minus"] = False

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

mpl.figure()

class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress', 
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(16):
    index = np.random.randint(1,60000)
    mpl.subplot(4,4,i+1)
    mpl.axis('off')
    if np.random.randint(0, 2) % 2:
        mpl.imshow(Image.fromarray(train_x[index]).transpose(Image.ROTATE_270), cmap='gray')
    else:
        mpl.imshow(Image.fromarray(train_x[index]).crop((0,0,20,20)), cmap='gray')
    mpl.title('标签值：'+ class_names[train_y[index]], fontsize=14)

mpl.suptitle('FASHION_MNIST测试集样本', fontdict={'color':'b'}, fontsize=40)
mpl.show()