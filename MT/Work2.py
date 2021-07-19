from PIL import Image
from PIL import ImageOps as ImgOp
from PIL import ImageFilter as ImgFlt
import matplotlib.pyplot as plt
import numpy as np

for i in range(0,10):
    img = Image.open(f'C:\\Users\小EK\Desktop\PyWork\MT\\number_data\{i}.png')
    img_r, img_g, img_b, a = img.split()

    # 图片反色

    img_inverse = ImgOp.invert(Image.merge('RGB',(img_r, img_g, img_b)))
    plt.imshow(img_inverse)

    # 高斯函数
    img_Gauss = img_inverse.filter(ImgFlt.GaussianBlur())
    plt.imshow(img_Gauss)

    # 二值化
    # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
    threshold = 100
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    # 'L'模式为灰色图像
    img_bin = img_Gauss.convert('L').point(table, '1')
    plt.imshow(img_bin)

    # 缩放
    img_size28 = img_bin.resize((28,28))
    plt.imshow(img_size28)

    # 转化为数组
    print(np.array(img_size28,dtype=int))