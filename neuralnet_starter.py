import numpy as np
import pickle


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 10  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  return probs


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  data = pickle.load(open(fname, "rb"))

  images = data[:, :-1]
  y = data[:, -1].astype(int)
  labels = np.zeros((y.shape[0], 10))
  labels[np.arange(y.shape[0]), y] = 1

  return images, labels


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None
    # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.relu(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = 1 / (1 + np.exp(-x))
    return output

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.tanh(x)

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = (x > 0) * x
    return output

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    return 1 - self.tanh(self.x) ** 2

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    return (self.x >= 0)


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    self.a = np.dot(self.x, self.w) + self.b
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    self.d_w = np.dot(self.x.T, delta)  # dy/dw = x
    # debug message
    # print('x', self.x.shape)
    # print('d_w', self.d_w.shape)

    self.d_b = delta.sum(axis=0)  # dy/db = 1
    # debug message
    # print('d_b', self.d_b.shape)
    # print('b', self.b.shape)

    self.d_x = np.dot(delta, self.w.T)  # dy/dx = w
    # debug message
    # print('w', self.w.shape)

    return self.d_x

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.targets = targets
    self.x = x
    self.y = x
    for i in self.layers:
      self.y = i.forward_pass(self.y)
    self.y = softmax(self.y) # softmax at last to calculate distributions. 

    if targets.any() == None:
      loss = None
    else:
      loss = self.loss_func(self.y, targets)
    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    output = -np.sum(targets * np.log(logits + 0.00001))# / logits.shape[0]
    return output
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    if self.targets.any() == None:
      return

    delta = (self.targets - self.y) #/ self.y.shape[0]
    for i in range(len(self.layers)-1, -1, -1):
      delta = self.layers[i].backward_pass(delta)
    
      

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  batch_size = config['batch_size']
  lr = config['learning_rate']
  penalty = config['L2_penalty']
  momentum = 0 # Momentum.
  val_loss_inc = 0
  last_loss_valid = None
  for i in range(config['epochs']):
    for t in range(int(X_train.shape[0]/batch_size)):
      batch_X = X_train[t * batch_size: (t+1)*batch_size]
      batch_Y = y_train[t * batch_size: (t + 1) * batch_size]

      loss_train, _ = model.forward_pass(batch_X, batch_Y)
      model.backward_pass()
      templayer = Layer(1,1)
      for layer in model.layers:
        if type(layer) == type(templayer):
          if config['momentum']:
            # Momentum update
            gamma = config['momentum_gamma']
            if not hasattr(layer, 'v'):
              layer.v = np.zeros_like(layer.w)

            # w
            d_w = gamma * layer.v + layer.d_w * lr
            layer.w = layer.w * (1 - lr * penalty/batch_size) + d_w
            layer.v = d_w

            # b
            layer.b = layer.d_b * lr
          else:
            layer.w = layer.w * (1 - lr * penalty/batch_size) + layer.d_w * lr
            layer.b = layer.d_b * lr
    loss_valid, _ = model.forward_pass(X_valid,y_valid)
    print('epoch:', i, 'train loss:', loss_train, 'valid loss:', loss_valid)
    
    if last_loss_valid == None:
      last_loss_valid = loss_valid
    else:
      if last_loss_valid < loss_valid:
        val_loss_inc += 1
      else:
        val_loss_inc = 0
      last_loss_valid = loss_valid
      if val_loss_inc >= config['early_stop_epoch']:
        break

def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """

  loss, y_out = model.forward_pass(X_test, y_test)
  y_out = np.argmax(y_out, axis=1)[:, np.newaxis]
  sum = 0
  for i in range(len(y_out)):
    if y_test[i][y_out[i, 0]] == 1:
      sum += 1
  accuracy = sum/len(y_out)

  return accuracy
      

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)
  print(test_acc)
  # Gradient check. 
  