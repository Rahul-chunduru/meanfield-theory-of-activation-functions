"""
Documentation

Generic layer. 

units - no. of nodes in the layer.
activation - type of activation

Builds weights lazily on given input.

Model.

"""

class GenericLayer(tf.keras.layers.Layer):
  def __init__(self, units, activation, name=None): 
     super(GenericLayer, self).__init__(name=name) 
     self.units = units
     self.activation = activation
  
  def build(self, input_shape):
      self.w = self.add_weight( 
        shape=([input_shape[1], self.units]), initializer='random_normal', name='w')
      self.beta = self.add_weight( 
        shape=([input_shape[1], self.units]), initializer='random_normal', name='beta')
      self.b = self.add_weight( 
        shape=([self.units]), initializer='random_normal', name='b')

  def call(self, x): 
     y = tf.matmul(x, self.w) + self.b 
     if self.activation == 'esp':
        y = tf.matmul(y, tf.math.sigmoid(tf.mathmul(self.beta, y)))
     elif self.activation == 'relu':
        y = tf.nn.relu(y)
     elif self.activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
     
     return y 


class DNN(tf.keras.Model):

    def __init__(self, layers, activation):
        super(DNN, self).__init__()
        self.Layers = [GenericLayer(layers[i], activation[i]) for i in range(len(layers))]

    def call(self, inputs):

        outputs = inputs
        for layer in self.Layers:
          outputs = layer(outputs)

        return outputs

model = DNN([2, 1], ['sigmoid', 'relu'])

x = tf.ones((100, 4)).numpy()
n1 = GenericLayer(4, 2, ['relu'])
n2 = GenericLayer(2, 1, ['relu'])

model(x)



