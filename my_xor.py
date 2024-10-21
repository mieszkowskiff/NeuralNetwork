import numpy as np
from icecream import ic
import copy
from neural_network import NeuralNetwork

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Net:
    def __init__(self):
        self.weights1 = np.random.rand(2, 2)
        self.bias1 = np.random.rand(2, 1)

        self.weights2 = np.random.rand(1, 2)
        self.bias2 = np.random.rand(1, 1)

    def forward(self, x):
        self.z1 = np.matmul(self.weights1, x) + self.bias1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.matmul(self.weights2, self.a1) + self.bias2
        self.a2 = sigmoid(self.z2)

        return self.a2
    
    def backward(self, x, y, learning_rate):
        d_a2 = self.a2 - y
        d_z2 = d_a2 * sigmoid_derivative(self.z2)
        d_weights2 = np.matmul(d_z2, self.a1.T)
        d_bias2 = d_z2

        d_a1 = np.matmul(self.weights2.T, d_z2)
        d_z1 = d_a1 * sigmoid_derivative(self.z1)

        d_weights1 = np.matmul(d_z1, x.T)
        d_bias1 = d_z1

        self.weights1 -= d_weights1 * learning_rate
        self.bias1 -= d_bias1 * learning_rate

        self.weights2 -= d_weights2 * learning_rate
        self.bias2 -= d_bias2 * learning_rate
       
def assert_equal(net1, net2):
    assert np.allclose(net1.weights1, net2.weights[0])
    assert np.allclose(net1.bias1, net2.biases[0])
    assert np.allclose(net1.weights2, net2.weights[1])
    assert np.allclose(net1.bias2, net2.biases[1])


np.random.seed(45)


net = Net()
net2 = NeuralNetwork([2, 2, 1])



#net2.weights = copy.deepcopy([net.weights1, net.weights2])
#net2.biases = copy.deepcopy([net.bias1, net.bias2])



x = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
y = np.array([[0], [1], [1], [0]])

for i in range(10000):
    for j in range(4):
        net2.backward(x[j], y[j], 0.5)

for j in range(4):
    ic(net2(x[j]))

for i in range(10000):
    for j in range(4):
        net.forward(x[j])
        net.backward(x[j], y[j], 0.5)

for j in range(4):
    ic(net.forward(x[j]))

    