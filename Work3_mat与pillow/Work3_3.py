''' 图像基本操作 '''

import matplotlib.pyplot as mpl
from PIL import Image

mpl.rcParams["font.sans-serif"] = ['SimHei']
mpl.rcParams["axes.unicode_minus"] = False

img = Image.open('Work3_mat与pillow\lena.tiff')
img_r, img_g, img_b = img.split()

print(img.size)
# ir,_,_ = img.resize(50, 50).split()

mpl.figure(figsize=(10,10))

mpl.subplot(221)
mpl.axis('off')
mpl.imshow(img_r.resize((50,50)), cmap='gray')
mpl.title('R-缩放',fontsize=14)

mpl.subplot(222)
mpl.axis('off')
mpl.imshow(img_g.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270), cmap='gray')
mpl.title('G-镜像+旋转',fontsize=14)

mpl.subplot(223)
mpl.axis('off')
mpl.imshow(img_b.crop((0,0,300,300)), cmap='gray')
mpl.title('B-裁剪',fontsize=14)

mpl.subplot(224)
mpl.axis('off')
mpl.imshow(Image.merge('RGB',(img_r,img_g,img_b)), cmap='gray')
Image.merge('RGB',(img_r,img_g,img_b)).save('Work3_mat与pillow\\test.png')
mpl.title('RGB',fontsize=14)

mpl.suptitle('图像基本操作', fontdict={'color':'b'}, fontsize=20)

mpl.show()