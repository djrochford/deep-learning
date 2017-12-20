import numpy as np

class _Layer:
  
  def __init__(self, activation_index, previous_dimension, dimension, normalization):
    self.activation = lookup_activation(activation_index)
    self.dimension = dimension
    self.weights = np.random.randn(dimension, previous_dimension) * normalization
    self.bias = np.zeros((dimension, 1))

  def lookup_activation(self, activation_index):
    activation_dict = {
      "relu": lambda Z: max(Z, 0)
      "sigmoid": lambda Z: 1/(1+np.exp(-Z))
      "tanh": lambda Z: np.tanh(Z)
    }
    return activation_dict[activation_index]

  def forward(self, inputs):
    assert(inputs.shape[0] == previous_dimension)
    Z = np.dot(self.weights, inputs) + bias
    return self.activation(Z)

class Neural_Net:

  def __init__(self, input_dimension, layer_definitions, normalization = 0.01):
    self.input_dimension
    self.normalization = normalization
    self.layers = self.initialize_layers(layer_definitions)

  def initialize_layers(self):
    layers = [None]
    previous_dimension = self.input_dimension
    for definition in layer_definitions:
      activation = layer_definition[0]
      dimension = layer_definition[1]
      layer = _Layer(activation, previous_dimension, dimension, normalization)
      layers.append(layer)
      previous_dimension = dimension

  def forward(self, X):
    inputs = X
    for layer in layers:
      output = layer.forward(inputs)
      inputs = output
    return output




