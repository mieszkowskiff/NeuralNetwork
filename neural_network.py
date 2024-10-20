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

        self.calculate_chain(input, output)

        weights_gradient[0] = np.matmul(self.chain[0], self.activation(input).T)
        biases_gradient[0] = self.chain[0]
        for i in range(1, self.layers):
            weights_gradient[i] = np.matmul(self.chain[i], self.activation(self.neurons[i - 1]).T)
            biases_gradient[i] = self.chain[i]
        
        for i in range(self.layers):
            self.weights_gradient[i] += weights_gradient[i]
            self.biases_gradient[i] += biases_gradient[i]

    def end_batch(self, learn_rate = 1):
        for layer in range(self.layers):
            self.weights[layer] += self.weights_gradient[layer] * learn_rate * -1
            self.biases[layer] += self.biases_gradient[layer] * learn_rate * -1

        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradinet = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]



if __name__ == "__main__":
    np.random.seed(0)
    nn = NeuralNetwork([2, 2, 1])
    
    nn.weights = [np.array([[2, 2], [1, 1]]), np.array([[-2, 1]])]
    nn.biases = [np.array([[-3], [0]]), np.array([[0]])]

    print(nn(np.array([[0], [0]])))

    

    


                
                    




        
        
