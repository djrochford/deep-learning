import unittest
import numpy as np
from neural_net import _Layer, Neural_Net

class Test_Layer(unittest.TestCase):

  def setUp(self):
    np.random.seed(1)
    my_layer = None
    self.array1 = np.array([[1, -1, 0], [2, 0.3, -7]])

  def test_instantiation(self):
    my_layer = _Layer('relu', 4, 3, 0.01)
    self.assertEqual(my_layer.activation(self.array1).tolist(), [[1, 0, 0],[2, 0.3, 0]])
    self.assertEqual(my_layer.backward_activation(self.array1).tolist(), [[1, 0, 0], [1, 1, 0]])
    self.assertEqual(my_layer.dimension, 3)
    self.assertEqual(my_layer.weights.shape, (3, 4))
    self.assertEqual(my_layer.bias.shape, (3, 1))
    self.assertEqual(my_layer.weights_gradient, None)
    self.assertEqual(my_layer.bias_gradient, None)
    self.assertEqual(my_layer.inputs, None)

  def test_lookup_activation(self):
    sigmoid = lambda x: 1/(1+np.exp(-x))
    my_layer = _Layer('sigmoid', 7, 2, 0.01)
    sigmoided_list = [[sigmoid(x) for x in row] for row in self.array1.tolist()]
    self.assertEqual(my_layer.activation(self.array1).tolist(), sigmoided_list)
    self.assertEqual(my_layer.backward_activation(np.array(sigmoided_list)).tolist(), [[x * (1 - x) for x in row] for row in sigmoided_list])
    my_other_layer = _Layer('tanh', 12, 10, 0.05)
    tanhed_list = [[np.tanh(x) for x in row] for row in self.array1.tolist()]
    self.assertEqual(my_other_layer.activation(self.array1).tolist(), tanhed_list)
    self.assertEqual(my_other_layer.backward_activation(np.array(tanhed_list)).tolist(), [[1 - x**2 for x in row] for row in tanhed_list])

  def test_forward(self):
    my_layer = _Layer('relu', 7, 8, 0.02)
    with self.assertRaises(AssertionError):
        my_layer.forward(np.array([[1],[2],[3],[4],[5],[6],[7],[8]]))
    inputs = np.array([[1],[2],[3],[4],[5],[6],[7]])
    output = my_layer.forward(inputs)
    self.assertTrue(output.shape == (8, 1))
    relu = lambda Z: np.maximum(Z, 0)
    np.random.seed(1)
    self.assertEqual(list(relu(np.dot(my_layer.weights, inputs))),
      list(output))
  
  def test_backward(self):
    my_layer = _Layer('sigmoid', 5, 3, 0.05)
    with self.assertRaises(AssertionError):
      my_layer.backward(np.array([[1],[-3],[0.04],[3],[-0.2]]))
    my_layer.forward(np.array([[1], [0], [4], [2], [0.3]]))
    assert not my_layer.inputs is None
    with self.assertRaises(AssertionError):
      my_layer.backward(np.array([[0.3], [2]]))
    input_gradient = my_layer.backward(np.array([[1], [-0.5], [0.44]]))
    self.assertEqual(input_gradient.shape, (5, 1))

  def test_update_parameters(self):
    my_layer = _Layer('tanh', 3, 3, 0.01)
    with self.assertRaises(AssertionError):
      my_layer.update_parameters(0.1)
    my_layer.forward(np.array([[2.2],[2.7],[1]]))
    my_layer.backward(np.array([[0.5], [4], [-1]]))
    old_weights = my_layer.weights
    old_bias = my_layer.bias
    my_layer.update_parameters(0.1)
    self.assertTrue(np.array_equal(my_layer.weights, old_weights - 0.1 * my_layer.weights_gradient))
    self.assertTrue(np.array_equal(my_layer.bias, old_bias - 0.1 * my_layer.bias_gradient))

class Test_Network(unittest.TestCase):

  def test_instantiation(self):
    my_network = Neural_Net(10, [('relu', 5)] * 7 + [('sigmoid', 1)])
    self.assertEqual(my_network.input_dimension, 10)
    self.assertEqual(my_network.normalization, 0.01)
    self.assertEqual(my_network.learning_rate, 0.1)
    self.assertEqual(len(my_network.layers), 9)
    self.assertEqual(len(my_network.parameters), 9)

  def test_forward(self):
    my_network = Neural_Net(10, [('relu', 5)] * 3 + [('sigmoid', 1)])
    inputs = [[1, 5], [2, 0], [3, -2], [-4, 1], [0.5, 0], [-0.2, 5], [2, -1], [2, -0.43], [0, 1.12], [0, -2]]
    output = my_network._forward(np.array(inputs))
    self.assertEqual(output.shape, (1,2))
    relu = lambda Z: np.maximum(Z, 0)
    sigmoid = lambda Z: 1/(1+np.exp(-Z))
    first = relu(np.dot(my_network.parameters[1][0], inputs) + my_network.parameters[1][1])
    second = relu(np.dot(my_network.parameters[2][0], first) + my_network.parameters[2][1])
    third = relu(np.dot(my_network.parameters[3][0], second) + my_network.parameters[3][1])
    expected = sigmoid(np.dot(my_network.parameters[4][0], third) + my_network.parameters[4][1])
    self.assertTrue(np.array_equal(expected, output))

  def test_calculate_cost(self):
    my_network = Neural_Net(10, [('relu', 5)] * 3 + [('sigmoid', 1)])
    inputs = [[1, 5], [2, 0], [3, -2], [-4, 1], [0.5, 0], [-0.2, 5], [2, -1], [2, -0.43], [0, 1.12], [0, -2]]
    output = my_network._forward(np.array(inputs))
    Y = np.array([[1, 0]])
    cost = my_network._calculate_cost(output, Y)
    expected = np.squeeze(-1/2 * np.dot(np.log(output), Y.T) + np.dot(np.log(1-output), (1-Y.T)))
    self.assertTrue(np.array_equal(expected, cost))

  def test_backward(self):
    my_network = Neural_Net(10, [('relu', 5)] * 3 + [('sigmoid', 1)])
    inputs = [[1, 5], [2, 0], [3, -2], [-4, 1], [0.5, 0], [-0.2, 5], [2, -1], [2, -0.43], [0, 1.12], [0, -2]]
    output = my_network._forward(np.array(inputs))
    Y = np.array([[1, 0]])
    antirelu = lambda A: (A > 0) * 1 + (A <= 0) * 0
    antisigmoid = lambda A: A * (1 - A)
    output_gradient = - ((Y / output) - ((1-Y) / (1-output)))
    fourth_linear_gradient = output_gradient * antisigmoid(my_network.layers[4].linear_output)
    fourth_weights_gradient = (1/2) * np.dot(fourth_linear_gradient, my_network.layers[4].inputs.T)
    fourth_bias_gradient = (1/2) * np.sum(fourth_linear_gradient, axis=1, keepdims=True)
    third_output_gradient = np.dot(my_network.layers[4].weights.T, fourth_linear_gradient)
    third_linear_gradient = third_output_gradient * antirelu(my_network.layers[3].linear_output)
    third_weights_gradient = (1/2) * np.dot(third_linear_gradient, my_network.layers[3].inputs.T)
    third_bias_gradient = (1/2) * np.sum(third_linear_gradient, axis=1, keepdims=True)
    second_output_gradient = np.dot(my_network.layers[3].weights.T, third_linear_gradient)
    second_linear_gradient = second_output_gradient * antirelu(my_network.layers[2].linear_output)
    second_weights_gradient = (1/2) * np.dot(second_linear_gradient, my_network.layers[2].inputs.T)
    second_bias_gradient = (1/2) * np.sum(second_linear_gradient, axis=1, keepdims=True)
    first_output_gradient = np.dot(my_network.layers[2].weights.T, second_linear_gradient)
    first_linear_gradient = first_output_gradient * antirelu(my_network.layers[1].linear_output)
    first_weights_gradient = (1/2) * np.dot(first_linear_gradient, my_network.layers[1].inputs.T)
    first_bias_gradient = (1/2) * np.sum(first_linear_gradient, axis=1, keepdims=True)
    my_network._backward(output, Y)
    self.assertTrue(np.array_equal(fourth_weights_gradient, my_network.layers[4].weights_gradient))
    self.assertTrue(np.array_equal(fourth_bias_gradient, my_network.layers[4].bias_gradient))
    self.assertTrue(np.array_equal(third_weights_gradient, my_network.layers[3].weights_gradient))
    self.assertTrue(np.array_equal(third_bias_gradient, my_network.layers[3].bias_gradient))
    self.assertTrue(np.array_equal(second_weights_gradient, my_network.layers[2].weights_gradient))
    self.assertTrue(np.array_equal(second_bias_gradient, my_network.layers[2].bias_gradient))
    self.assertTrue(np.array_equal(first_weights_gradient, my_network.layers[1].weights_gradient))
    self.assertTrue(np.array_equal(first_bias_gradient, my_network.layers[1].bias_gradient))

unittest.main()