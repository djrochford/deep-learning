import numpy as np
from itertools import islice

class _Layer:
  
  def __init__(self, activation_index, previous_dimension, dimension, normalization):
    self.activation, self.backward_activation = self.lookup_activation(activation_index)
    self.dimension = dimension
    self.weights = np.random.randn(dimension, previous_dimension) * normalization
    self.bias = np.zeros((dimension, 1))
    self.inputs = None
    self.linear_output = None
    self.weights_gradient = None
    self.bias_gradient = None

  def lookup_activation(self, activation_index):
    activation_dict = {
      "relu": (lambda Z: np.maximum(Z, 0), lambda A: (A > 0) * 1 + (A <= 0) * 0),
      "sigmoid": (lambda Z: 1/(1+np.exp(-Z)), lambda A: A * (1 - A)),
      "tanh": (lambda Z: np.tanh(Z), lambda A: 1 - np.power(A, 2))
    }
    return activation_dict[activation_index]

  def forward(self, inputs):
    assert inputs.shape[0] == self.weights.shape[1]
    self.inputs = inputs
    self.linear_output = np.dot(self.weights, self.inputs) + self.bias
    return self.activation(self.linear_output)

  def backward(self, output_gradient):
    assert not self.inputs is None
    assert not self.linear_output is None
    assert output_gradient.shape[0] == self.linear_output.shape[0]
    linear_gradient = output_gradient * self.backward_activation(self.linear_output)
    m = self.inputs.shape[1]
    self.weights_gradient = (1/m) * np.dot(linear_gradient, self.inputs.T)
    self.bias_gradient = (1/m) * np.sum(linear_gradient, axis=1, keepdims=True)
    input_gradient = np.dot(self.weights.T, linear_gradient)
    return input_gradient

  def update_parameters(self, learning_rate):
    assert not self.weights_gradient is None
    assert not self.bias_gradient is None
    self.weights = self.weights - learning_rate * self.weights_gradient
    self.bias = self.bias - learning_rate * self.bias_gradient

class Neural_Net:

  def __init__(self, input_dimension, layer_definitions, normalization = 0.01, learning_rate = 0.1):
    self.input_dimension = input_dimension
    self.normalization = normalization
    self.learning_rate = learning_rate
    self.layers, self.parameters = self._initialize_layers(layer_definitions)

  def _initialize_layers(self, layer_definitions):
    layers = [None]
    parameters = [None]
    previous_dimension = self.input_dimension
    for definition in layer_definitions:
      activation = definition[0]
      dimension = definition[1]
      layer = _Layer(activation, previous_dimension, dimension, self.normalization)
      layers.append(layer)
      parameters.append((layer.weights, layer.bias))
      previous_dimension = dimension
    return layers, parameters

  def _forward(self, X):
    assert X.shape[0] == self.input_dimension
    inputs = X
    for layer in islice(self.layers, 1, None):
      output = layer.forward(inputs)
      inputs = output
    return output

  def _calculate_cost(self, output, Y):
    assert output.shape == Y.shape
    m = Y.shape[1]
    cost = (-1/m) * np.dot(np.log(output), Y.T) + np.dot(np.log(1-output), (1-Y).T)
    cost = np.squeeze(cost)
    return cost

  def _backward(self, output, Y):
    output_gradient = - ((Y / output) - ((1-Y) / (1-output)))
    for i in range(len(self.layers) - 1, 0, -1):
      output_gradient = self.layers[i].backward(output_gradient)

  def _update_parameters(self):
    for layer in islice(layers, 1, None):
      layer.update_parameters(self.learning_rate)


