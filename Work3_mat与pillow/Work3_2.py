import matplotlib.pyplot as mpl
import numpy as np
import tensorflow as tf

boston_house = tf.keras.datasets.boston_housing

(test_x, test_y), (_, _)=boston_house.load_data(test_split=0)

mpl.rcParams["font.sans-serif"] = ['SimHei']
mpl.rcParams["axes.unicode_minus"] = False

titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
            "TAX", "PTRATIO", "B-1000", "LSTAT"]

fig = mpl.figure()
choice = 0;

mpl.suptitle("请输入属性（#16进制）", x=0.5, y=0.99, fontsize=20)
mpl.axis('off')
for i in range(13):
    mpl.text(0.5,1-i*0.05,str(i+1)+'.'+titles[i])

# 时间event获取键盘输入
def onscanf(event):
    choice = int(event.key, 16) # 键盘输入属性选择序号
    
    mpl.clf()
    mpl.subplot()
    mpl.scatter(test_x[:,choice], test_y)
    mpl.xlabel(titles[choice])
    mpl.ylabel('Price($1000\'s)')
    mpl.title(str(choice+1)+'.'+titles[choice]+"-Price")
    mpl.subplots_adjust(top=0.90)
    mpl.tight_layout()
    mpl.suptitle("各属性与房价的关系", x=0.5, y=0.99, fontsize=20)
    mpl.subplots_adjust(top=0.88)
    mpl.draw()

cid = fig.canvas.mpl_connect('key_press_event', onscanf)

mpl.show()
