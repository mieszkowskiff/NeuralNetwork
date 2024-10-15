import numpy as np
import math as m
import copy

def sigmoid(x):
    if x > 0:
        if x > 700:
            return 1
        exp = m.exp(x)
        return exp / (1 + exp)
    if x < -700:
        return 0
    return 1 / (1 + m.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

    
class NeuralNetwork:
    def __init__(self, structure):
        self.structure = np.array(structure)
        self.layers = self.structure.shape[0] - 1

        self.weights = [np.random.rand(self.structure[i + 1], self.structure[i]) for i in range(self.layers)]
        self.biases = [np.random.rand(self.structure[i + 1], 1) for i in range(self.layers)]

        self.activation = np.vectorize(sigmoid)
        self.activation_derivative = np.vectorize(sigmoid_derivative)

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

    def calculate_cost(self, input, output):
        self.assure_input(input)
        self.assure_output(output)

        return sum((self(input) - output)**2)[0]


    def calculate_neurons(self, input):
        self.assure_input(input)
        self.neurons[0] = np.matmul(self.weights[0], self.activation(copy.deepcopy(input))) + self.biases[0]
        for i in range(1, self.layers):
            self.neurons[i] = np.matmul(self.weights[i], self.activation(self.neurons[i - 1])) + self.biases[i]

    
    def calculate_chain(self, input, output):
        self.calculate_neurons(input)
        self.assure_output(output)
        self.chain[-1] = 2 * (self.activation(self.neurons[-1]) - output) * self.activation_derivative(self.neurons[-1])
        for i in range(self.layers - 2, -1, -1):
            self.chain[i] = np.matmul(self.weights[i + 1].T, self.chain[i + 1]) * self.activation_derivative(self.neurons[i])
    
    def calculate_gradient(self, input, output):
        self.assure_input(input)
        self.assure_output(output)

        weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.calculate_neurons(input)
        self.calculate_chain(input, output)

        for layer in range(self.layers):
            for i in range(self.structure[layer + 1]):
                for j in range(self.structure[layer]):
                    weights_gradient[layer][i][j] = self.chain[layer][i][0] * self.neurons[layer][j][0]
                biases_gradient[layer][i][0] = self.chain[layer][i][0]
        
        for layer in range(self.layers):
            self.weights_gradient[layer] += weights_gradient[layer]
            self.biases_gradient[layer] += biases_gradient[layer]


    def end_batch(self, learn_rate = 1):
        for layer in range(self.layers):
            self.weights[layer] += self.weights_gradient[layer] * learn_rate * -1
            self.biases_gradient[layer] += self.biases_gradient[layer] * learn_rate * -1

        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradinet = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]



if __name__ == "__main__":
    print(np.array([1, 2, 3]) * np.array([1, 2, 3]))
    
    
    

    


                
                    




        
        
