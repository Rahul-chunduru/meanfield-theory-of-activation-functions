import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

m, n, o = x_train.shape

x_train = x_train.reshape(m, n * o)
y_train = one_hot_econding(y_train, 10, 0)