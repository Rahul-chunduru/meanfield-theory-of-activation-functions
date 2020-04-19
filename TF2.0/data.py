import tensorflow as tf
import numpy as np

def oneHotEncode(y):

    num = len(y)
    nlabels = len(np.unique(y))

    one_hot_matrix = np.zeros((num, nlabels))

    for i, c in enumerate(y):
        one_hot_matrix[i][y[i] - 1] = 1

    return one_hot_matrix

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

m, n, o = x_train.shape

x_train = x_train.reshape(m, n * o)
y_train = oneHotEncode(y_train)