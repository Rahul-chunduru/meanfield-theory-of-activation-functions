M = GenericLayer(2, 1, ['sigmoid'])
x = tf.ones((100, 2))


with tf.GradientTape() as g:
  z = tf.norm(M(x))

grads = g.gradient(z, M.trainable_variables)
tf.print(grads[0])