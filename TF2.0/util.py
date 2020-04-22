  #Math Libraries
import numpy as np

#Visualization libraries
import matplotlib.pyplot as plt

#Tensor Flow
import tensorflow as tf

from tensorflow.python.framework import ops

tf.keras.backend.set_floatx('float64')

class GenericLayer(tf.keras.layers.Layer):
  def __init__(self, units, activation, name=None): 
     super(GenericLayer, self).__init__(name=name) 
     self.units = units
     self.activation = activation
  
  def build(self, input_shape):
      
      train = True
      if self.activation != 'esp':
        train = False

      self.w = self.add_weight( 
        shape=([input_shape[1], self.units]), initializer='random_normal', name='w', dtype='float64')
      self.beta = self.add_weight( 
        shape=([self.units]), initializer='random_normal', name='beta', dtype='float64', trainable=train)
      self.b = self.add_weight( 
        shape=([self.units]), initializer='random_normal', dtype='float64', name='b')

  def call(self, x): 
     y = tf.matmul(x, self.w) + self.b 
     if self.activation == 'esp':
        y = tf.math.multiply(y, tf.math.sigmoid(tf.math.multiply(self.beta, y)))    
     elif self.activation == 'relu':
        y = tf.nn.relu(y)
     elif self.activation == 'sigmoid':
        y = tf.nn.sigmoid(y)
     elif self.activation == 'softmax':
        y = tf.nn.softmax(y)
     
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

def obj(y_true, y_pred, method):

    """
    Arguments:
    y_true, y_pred

    Returns:
    cost -- cost function

    """

    if method == 'crossEntropy': #use cross entropy loss function
          cce = tf.keras.losses.CategoricalCrossentropy() 
          cost  = cce(y_true, y_pred)

    elif method == 'meanSquared': #use minimum squared error (L2 loss)
          cost = tf.keras.losses.MSE(y_true, y_pred)

    return cost

def eval_accuracy(y_true, y_pred, method):
  # y dimension [batch_size x N_classes]

  accuracy = 0

  if method == 'logistic': #accuracy score for binary class classification
      correct = tf.equal(tf.argmax(y_pred, axis = 1), tf.argmax(y_true, axis = 1))
      accuracy= tf.reduce_mean(tf.cast(correct, "float64"))

  elif method == 'regression': #r2 score
      norm= tf.reduce_mean( tf.squared_difference( y_true,tf.reduce_mean(y_true)) )
      accuracy = 1 - tf.divide( tf.reduce_mean(tf.squared_difference(y_pred, y_true)), norm)
  
  return accuracy

#------------Hessian-------------------

def flatten(tensor):

    '''
    Flattening function:

    input: a tensor list
    returns: a rank one tensor
    '''

    s= len(tensor) #number of tensors in the list

    for i in range(s):

        dl = tensor[i] #take one element of the gradient list (hence the zero)
        d1, d2 = dl.get_shape() #Obtain tensor dimensions

        fl = tf.reshape(dl,[-1, d1*d2]) #reshape the tensor to a (1, d1*d2) tensor

        #concatenate over all the elemets in the list
        if i==0: flattened = fl # the first time
        else: flattened = tf.concat([flattened, fl], axis=1)

    return flattened

#Hessian
def hessian(grads, par):

    '''
    Evaluates the exact Hessian matrix.
    This function uses the same convention of the Autograd package.

    Inputs:
    grads --- the evaluated gradeints of the cost function

    Returns:
    hessian matrix: a (dim,dim) matrix of second derivatives, where 'dim' is the dimension of
    the flattened gradient tensor.
    '''

    flat_grads = flatten(grads)[0] #flat gradients

    dim = flat_grads.get_shape()[0] #get the dimensions of the flattened tensor

    hess = [] #list
    for i in range (dim):

        dg_i = tf.gradients(flat_grads[i], par) #for each element of grads evaluate the gradients
        dg_i_flat = flatten(dg_i) #flatten the resulting hessian onto a 1 d array
        hess.append(dg_i_flat) #store row by row

    return tf.reshape(hess,[dim, dim]) #returns the reshaped matrix

def train(x, y, model, obj, method, num_iter, lr, tracV, sEpoch):

  opt = tf.keras.optimizers.SGD(lr)
  
  tracD = {}
  for var in tracV:
    tracD[var] = []

  for epoch in range(1, num_iter + 1):
    with tf.GradientTape() as g:
      cost = obj(model(x), y, method)
      print("epoch", epoch, cost)

    grads = g.gradient(cost, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % sEpoch == 0:
      for var in tracV:
        tracD[var].append(model.trainable_variables[var].numpy())
    
  for var in tracV:
    tracD[var] = np.stack(tracD[var])
   
  return model, tracD

# sample code
#plotting code

def plotNode(betaL, layerNum, nodeNum):
  for var in [4]:
    tracVar = betaL[layerNum]
    x = np.linspace(100, 3000, 30)
    y = tracVar[:, nodeNum]
    # plt.title('')
    plt.xlabel('iterations')
    plt.ylabel('beta values')
    plt.plot(x, y)
    plt.show()