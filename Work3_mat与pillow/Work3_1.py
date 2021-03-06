import matplotlib.pyplot as plt
import numpy as np

area = [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21]
price = [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30]
figure = plt.figure()

plt.scatter(area, price, color='red')
plt.title("商品房销售记录", fontdict={'color':'b', 'fontsize':16, 'fontproperties':'SimSun'})
plt.xlabel('面积（平方米）', fontdict={'fontsize':14, 'fontproperties':'SimSun'})
plt.ylabel('价格（万元）', fontdict={'fontsize':14, 'fontproperties':'SimSun'})
plt.show()
