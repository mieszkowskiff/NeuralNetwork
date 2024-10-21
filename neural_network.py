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



    def assure_input(self, input):
        assert type(input) is np.ndarray
        assert input.shape[0] == self.structure[0]


    def assure_output(self, output):
        assert type(output) is np.ndarray
        assert output.shape[0] == self.structure[-1]


    def __call__(self, input):
        self.assure_input(input)

        data = copy.deepcopy(input)
        for i in range(self.layers):
            data = self.activation(np.matmul(self.weights[i], data) + self.biases[i])
        return data



    def forward(self, input):
        self.assure_input(input)
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
    
    def backward(self, input, output, learning_rate = 0.1):
        self.assure_input(input)
        self.assure_output(output)

        weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.calculate_chain(input, output)

        weights_gradient[0] = np.matmul(self.chain[0], input.T)
        biases_gradient[0] = self.chain[0]
        for i in range(1, self.layers):
            weights_gradient[i] = np.matmul(self.chain[i], self.activation(self.neurons[i - 1]).T)
            biases_gradient[i] = self.chain[i]


        for i in range(self.layers):
            self.weights[i] -= weights_gradient[i] * learning_rate
            self.biases[i] -= biases_gradient[i] * learning_rate






if __name__ == "__main__":
    np.random.seed(0)
    nn = NeuralNetwork([2, 2, 1])
    
    nn.weights = [np.array([[2, 2], [1, 1]]), np.array([[-2, 1]])]
    nn.biases = [np.array([[-3], [0]]), np.array([[0]])]

    print(nn(np.array([[0], [0]])))

    

    


                
                    




        
        
