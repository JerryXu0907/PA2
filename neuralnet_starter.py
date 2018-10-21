import numpy as np
import pickle


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

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
    self.a = np.multiply(self.w, self.x) + self.b
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    self.d_w = self.x * delta # dy/dw = x
    self.d_b = delta # dy/db = 1
    self.d_x = self.w * delta # dy/dx = w
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

    if targets == None:
      loss = None
    else:
      loss = self.loss_func(self.y, targets)
    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    output = -np.sum(targets * np.log(logits)) / logits.shape[0]
    return output
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    if self.targets == None:
      return

    delta = (self.y - self.targets) / self.y.shape[0]
    for layer in self.layer:
      delta = layer.backward_pass(delta)
    
      

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  nn = Neuralnetwork(config)

  batch_num = 0
  v = 0 # Momentum. 
  val_loss_inc = 0
  last_loss_valid = None
  for i in xrange(config['epochs']):
    batch_X = X_train[batch_num : batch_num + config['batch_size']]
    batch_y = y_train[batch_num : batch_num + config['batch_size']]
    batch_num += config['batch_size']

    # Forward pass. 
    loss, y_out = nn.forward_pass(batch_X, batch_y)
    # L2 Reg. 
    l2 = 0
    for (layer in nn.layers):
      l2 += np.sum(np.square(layer.w))
    l2 *= 0.5 * config['L2_penalty'] / batch_X.shape[0]
    loss += l2

    # Backprop. 
    nn.backward_pass()

    # Update. 
    lr = config['learning_rate']
    for layer in nn.layers:
      # Add l2 for dw. 
      dw = layer.d_w + config['L2_penalty'] * layer.w

      # Update rules. 
      if config['momentum']:
        # Momentum update
        gamma = config['momentum_gamma']
        if layer.v == None:
          layer.v = np.zeros_like(layer.w)

        # w
        layer.v = gamma * layer.v - lr * dw
        layer.w += v

        # b
        layer.b += lr * layer.d_b
      else:
        # Vanilla update
        layer.w += lr * dw
        layer.b += lr * layer.d_b

    # Validation loss. 
    loss_valid, _ = nn.forward_pass(X_valid, y_valid)

    # Early stop. 
    if config['early_stop']:
      if last_loss_valid == None:
        last_loss_valid = loss_valid
        continue

      if loss_valid >= last_loss_valid:
        val_loss_inc += 1
      else:
        val_loss_inc = 1
        
      last_loss_valid = loss_valid
      if val_loss_inc >= config['early_stop_epoch']:
        break
    
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """

  loss, y_out = nn.forward_pass(X_test, y_test)
  y_out = np.argmax(y_out, axis=1)[:, np.newaxis]
  accuracy = np.sum((y_out == y_test) * 1) / y_test.shape[0]

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

