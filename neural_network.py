import numpy as np
import math as m

def sigmoid(x):
    if x > 0:
        exp = m.exp(x)
        return exp / (1 + exp)
    return 1 / (1 + m.exp(-x))


class NeuralNetwork:
    def __init__(self, structure):
        self.structure = np.array(structure)
        self.layers = self.structure.shape[0] - 1

        self.weights = [np.random.rand(self.structure[i + 1], self.structure[i]) for i in range(self.layers)]
        self.biases = [np.random.rand(self.structure[i + 1], 1) for i in range(self.layers)]

        self.activation = np.vectorize(sigmoid)

        self.neurons = [np.random.rand(self.structure[i],1) for i in range(self.layers + 1)]
        self.chain = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradinet = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]



    def assure_input(self, input):
        assert type(input) is np.ndarray
        assert input.shape[0] == self.structure[0]


    def assure_output(self, output):
        assert type(output) is np.ndarray
        assert output.shape[0] == self.structure[-1]


    def __call__(self, input):
        self.assure_input(input)

        data = input
        for i in range(self.layers):
            data = self.activation(np.matmul(self.weights[i], data) + self.biases[i])
        return data

    def calculate_cost(self, input, output):
        self.assure_input(input)
        self.assure_output(output)

        return sum((self(input) - output)**2)[0]


    def calculate_neurons(self, input):
        self.neurons[0] = input
        for i in range(self.layers):
            self.neurons[i + 1] = self.activation(np.matmul(self.weights[i], self.neurons[i]) + self.biases[i])

    
    def calculate_chain(self, input, output):
        for i in range(self.structure[-1]):
            self.chain[-1][i][0] = self.neurons[-1][i][0] * (1 - self.neurons[-1][i][0]) * 2 * (self.neurons[-1][i][0] - output[i][0])
            
        for layer in range(self.layers - 2, -1, -1):
            for i in range(self.structure[layer + 1]):
                self.chain[layer][i][0] = self.neurons[layer + 1][i] * (1 - self.neurons[layer + 1][i]) * sum(self.chain[layer + 1][j][0] * self.weights[layer + 1][j][i] for j in range(self.structure[layer + 2]))

    
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




    
    

    


                
                    




        
        
