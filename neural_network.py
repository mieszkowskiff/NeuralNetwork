import numpy as np
import copy
from icecream import ic

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, structure):
        self.structure = np.array(structure)
        self.layers = self.structure.shape[0] - 1

        self.weights = [np.random.rand(self.structure[i + 1], self.structure[i]) for i in range(self.layers)]
        self.biases = [np.random.rand(self.structure[i + 1], 1) for i in range(self.layers)]

        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

        self.neurons = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]
        self.chain = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.batch_size = 0



    def assure_input(self, input):
        assert type(input) is np.ndarray
        assert input.shape[0] == self.structure[0]


    def assure_output(self, output):
        assert type(output) is np.ndarray
        assert output.shape[0] == self.structure[-1]


    def __call__(self, input):
        #self.assure_input(input)

        data = copy.deepcopy(input)
        for i in range(self.layers):
            data = self.activation(np.matmul(self.weights[i], data) + self.biases[i])
        return data



    def forward(self, input):
        #self.assure_input(input)
        self.neurons[0] = np.matmul(self.weights[0], input) + self.biases[0]
        for i in range(1, self.layers):
            self.neurons[i] = np.matmul(self.weights[i], self.activation(self.neurons[i - 1])) + self.biases[i]
        return self.activation(self.neurons[-1])

    
    def calculate_chain(self, input, output):
        self.forward(input)
        self.assure_output(output)
        self.chain[-1] = (self.activation(self.neurons[-1]) - output) * self.activation_derivative(self.neurons[-1])
        for i in range(self.layers - 2, -1, -1):
            self.chain[i] = np.matmul(self.weights[i + 1].T, self.chain[i + 1]) * self.activation_derivative(self.neurons[i])
    
    def backward(self, input, output):
        self.assure_input(input)
        self.assure_output(output)

        self.calculate_chain(input, output)

        self.weights_gradient[0] = np.matmul(self.chain[0], input.T)
        self.biases_gradient[0] = self.chain[0]
        for i in range(1, self.layers):
            self.weights_gradient[i] += np.matmul(self.chain[i], self.activation(self.neurons[i - 1]).T)
            self.biases_gradient[i] += self.chain[i]

        self.batch_size += 1

    def end_batch(self, learning_rate = 1):
        for i in range(self.layers):
            self.weights[i] -= self.weights_gradient[i] * learning_rate
            self.biases[i] -= self.biases_gradient[i] * learning_rate
        
        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]
        self.batch_size = 0






if __name__ == "__main__":
    np.random.seed(89)
    net = NeuralNetwork([2, 2, 1])

    x = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
    y = np.array([[0], [1], [1], [0]])

    for i in range(30000):
        net.backward(x[i % 4], y[i % 4])
        if i % 4 == 0: 
            net.end_batch(1)

    print(net(x))

    

    


                
                    




        
        
