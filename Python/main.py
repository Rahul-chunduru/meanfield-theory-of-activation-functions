from esp_utils import *
from mnist import MNIST

mndata = MNIST('./python-mnist/data')

images, labels = mndata.load_training()

# need to hard code Ys

images = np.array(images)
labels = oneHotEncode(labels)

print("shapes", images.shape, labels.shape)

params, grads, costs = Run_DNN(images, labels, [28, 10], ['sigmoid', 'sigmoid', 'sigmoid'], 0.01, 100)

print('basic test pass')