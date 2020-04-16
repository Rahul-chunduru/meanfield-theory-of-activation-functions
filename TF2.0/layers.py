class GenericLayer(tf.Module):
  def __init__(self, in_features, out_features, actD, name=None): 
     super(GenericLayer, self).__init__(name=name) 
     self.w = tf.Variable( 
       tf.random.normal([in_features, out_features]), name='w')
     self.beta = tf.Variable( 
       tf.random.normal([out_features]), name='beta') 
     self.b = tf.Variable(tf.zeros([out_features]), name='b')
     self.actD = actD
  def __call__(self, x): 
     y = tf.matmul(x, self.w) + self.b 
     if self.actD[0] == 'esp':
        y = tf.matmul(y, tf.math.sigmoid(tf.mathmul(self.beta, y)))
     elif self.actD[0] == 'relu':
        y = tf.nn.relu(y)
     elif self.actD[0] == 'sigmoid':
        y = tf.nn.sigmoid(y)
     
     return y 