import numpy as np
import neural_network
from icecream import ic
# Funkcje aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Definicja klasy sieci neuronowej
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Losowa inicjalizacja wag
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        # Losowa inicjalizacja biasów
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def feedforward(self, X):
        # Warstwa ukryta
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Warstwa wyjściowa
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backpropagation(self, X, y, output, learning_rate):
        # Obliczenie błędów
        error = y - output
        d_output = error * sigmoid_derivative(output)
        
        # Obliczenie błędu w warstwie ukrytej
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Aktualizacja wag i biasów
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_output += np.sum(d_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backpropagation(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss:.5f}')
    
    def predict(self, X):
        return self.feedforward(X)



# Inicjalizacja sieci neuronowej

np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2)
weights_hidden_output = np.random.rand(2, 1)
bias_hidden = np.random.rand(1, 2)
bias_output = np.random.rand(1, 1)

nn1 = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
nn2 = neural_network.NeuralNetwork([2, 2, 1])

nn1.weights_input_hidden = weights_input_hidden
nn1.weights_hidden_output = weights_hidden_output
nn1.bias_hidden = bias_hidden
nn1.bias_output = bias_output

nn2.weights = [weights_input_hidden.T, weights_hidden_output.T]
nn2.biases = [bias_hidden.T, bias_output.T]

print(nn1.predict(np.array([[0, 0]])))
print(nn2(np.array([[0], [0]])))

