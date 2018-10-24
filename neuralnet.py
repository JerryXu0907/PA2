import numpy as np
import pickle
import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [784, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 300  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
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
      return self.ReLU(a)
  
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
  last_loss_valid = np.array([])
  saved_weights = []
  saved_biases = []
  best_epoch = 0
  train_acc = []
  test_acc = []
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

    # Store training and testing accuracy for each epoch. 
    train_acc.append(test(model, X_train, y_train, config))
    test_acc.append(test(model, X_test, y_test, config))
    #print('epoch:', i, 'train loss:', loss_train, 'valid loss:', loss_valid)
    

    if config["early_stop"]:
      if last_loss_valid.shape[0] == 0:
        last_loss_valid = np.array([loss_valid])
      else:
        if np.sum((loss_valid < last_loss_valid) * 1) != last_loss_valid.shape[0]:
          last_loss_valid = np.append(last_loss_valid, loss_valid)
        else:
          last_loss_valid = np.array([loss_valid])

          # Save the network with best val loss. 
          saved_weights = []
          saved_biases = []
          for layer in model.layers:
            if not hasattr(layer, 'w'):
              continue
            
            # The saved weights and biases should simply be weights and biases for each layers. 
            # The number of layers should match the length of saved_weights and saved_biases. 
            # The saved_weights[0] is for the first layer, [1] is for the second layer, and so on. 
            # The same for saved_biases. 
            saved_weights.append(np.copy(layer.w))
            saved_biases.append(np.copy(layer.b))
          best_epoch = i
        
        if last_loss_valid.shape[0] > config['early_stop_epoch']:
          break
      
  model.saved_weights = saved_weights
  model.saved_biases = saved_biases
  model.best_epoch = best_epoch
  model.train_acc = train_acc
  model.test_acc = test_acc

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


test_data_fname = 'MNIST_test.pkl'

X_test, y_test = load_data(test_data_fname)

def solve_questions():
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)

  #############################################################################
  #################################### Question 3, Part c #####################
  #############################################################################
  print("\n\nQuestion 3, Part c")
  # Use 10-fold cross validation to find the best epoch number. 
  cross_val_idx = np.split(np.arange(X_train.shape[0]), 10)
  best_epoches = []
  best_weights = []
  best_biases = []
  print("## Epoch Data start:")
  for i in range(10):
    # Generate one fold of holdout. 
    X_holdout = X_train[cross_val_idx[i]]
    y_holdout = y_train[cross_val_idx[i]]

    # Generate one fold of train data. 
    train_idx = np.delete(np.arange(X_train.shape[0]), cross_val_idx[i])
    X = X_train[train_idx]
    y = y_train[train_idx]
    
    # Train the model with given data. 
    model = Neuralnetwork(config)
    trainer(model, X, y, X_holdout, y_holdout, config)
    test_acc = test(model, X_test, y_test, config)

    # Store results and weights. 
    best_epoches.append(model.best_epoch)
    best_weights.append(model.saved_weights)
    best_biases.append(model.saved_biases)

    ###### Make table with this data ######
    # Need to report a table with 10 numbers of epochs where the weights were best. 
    print("best epoch:", model.best_epoch, "test acc:", test_acc)
  print("## Epoch Data stop")

  ########################### The final network. 
  # Use the average of these epoches to train the whole dataset. 
  epoch_avg = sum(best_epoches) / len(best_epoches)
  config["epochs"] = int(epoch_avg)
  config["early_stop"] = False

  print("## Average epoch data:", epoch_avg)

  print("\n## Final model")
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)
  print("test acc:", test_acc)

  ############## Plot with this data ##############
  # Need to describe training procedure AND
  # plot training and testing accuracy vs. number of training epochs of SGD. 
  # This plot only needed for the final network. 

  plt.plot(model.train_acc, label="train acc")
  plt.plot(model.test_acc, label="test acc")
  plt.title('Final model with {0} training epochs, training and testing accuracy vs. number of training epoches'.format(int(epoch_avg)))
  plt.xlabel('number of training epochs')
  plt.ylabel('training and testing accuracy')
  plt.legend()
  plt.savefig('Q3Pc_Final Network.png', bbox_inches='tight')
  plt.clf()
  print('Q3Pc_Final Network.png generated')
  '''
  print("## Train accuracies start:")
  print(model.train_acc)
  print("## Train accuracies end")
  print("## Test accuracies start:")
  print(model.test_acc)
  print("## Test accuracies end")
  '''



  #############################################################################
  ########################### Question 3, Part d ##############################
  #############################################################################
  print("\n\nQuestion 3, Part d")
  config["epochs"] = int(config["epochs"] * 1.1)

  regs_exp = [1e-3, 1e-4]
  for reg in regs_exp:
    config["L2_penalty"] = reg
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)

    ############## Plot with this data ##############
    # Need to report training and testing accuracy vs. number of epoches of SGD
    '''
    print("\nModel trained with l2 reg of", reg)
    print("test acc:", test_acc)
    print("## Train accuracies start:")
    print(model.train_acc)
    print("## Train accuracies end")
    print("## Test accuracies start:")
    print(model.test_acc)
    print("## Test accuracies end")
    '''
    plt.plot(model.train_acc, label="train acc")
    plt.plot(model.test_acc, label="test acc")
    plt.title('Model with l2 reg of {0}, training and testing accuracy vs. number of training epoches'.format(reg))
    plt.xlabel('number of training epochs')
    plt.ylabel('training and testing accuracy')
    plt.legend()
    plt.savefig('Q3Pd_reg_{0}.png'.format(reg), bbox_inches='tight')
    plt.clf()
    print('Q3Pd_reg_{0}.png generated'.format(reg))



  #############################################################################
  ########################### Question 3, Part e ##############################
  #############################################################################
  print("\n\nQuestion 3, Part e")
  activations_exp = ["sigmoid", "ReLU"]
  for activation in activations_exp:
    config["activation"] = activation
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)

    ############## Plot with this data ##############
    # Need to report training and testing accuracy vs. number of epoches of SGD
    # Comment on the change of performance. 
    '''
    print("\nModel trained with activation:", activation)
    print("test acc:", test_acc)
    print("## Train accuracies start:")
    print(model.train_acc)
    print("## Train accuracies end")
    print("## Test accuracies start:")
    print(model.test_acc)
    print("## Test accuracies end")
    '''
    
    plt.plot(model.train_acc, label="train acc")
    plt.plot(model.test_acc, label="test acc")
    plt.title('Model with activation {0}, training and testing accuracy vs. number of training epoches'.format(activation))
    plt.xlabel('number of training epochs')
    plt.ylabel('training and testing accuracy')
    plt.legend()
    plt.savefig('Q3Pe_activation_{0}.png'.format(activation), bbox_inches='tight')
    plt.clf()
    print('Q3Pe_activation_{0}.png generated'.format(activation))
  
