import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PrtalWIdth', 'Species']
tf_iris = pd.read_csv(train_path, names=COLUMN_NAMES, header=0)
iris = np.array(tf_iris)

fig = mpl.figure('Iris Data', figsize=(15,15))
fig.suptitle("Anderson's iris Data Set\n(Bule->Setosa|Red->Versicolor|Green->Virginica)", fontsize=20)

for i in range(4):
    for j in range(4):
        mpl.subplot(4, 4, 4*i+(j+1))
        if(i==j):
            mpl.hist(iris[:,j], align='mid', edgecolor='k')
        else:
            mpl.scatter(iris[:,j], iris[:,i], c=iris[:,4], cmap='brg')
        if(i==0):
            mpl.title(COLUMN_NAMES[j])
        if(j==0):
            mpl.ylabel(COLUMN_NAMES[i])

mpl.show()