import numpy as np

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
        self.hidden_input = np.matmul(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Warstwa wyjściowa
        self.final_input = np.matmul(self.hidden_output, self.weights_hidden_output) + self.bias_output
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


np.random.seed(0)
# Zbiór danych dla bramki XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], 
              [1], 
              [1], 
              [0]])

# Inicjalizacja sieci neuronowej
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Trenowanie sieci
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Testowanie sieci
for input_data in X:
    prediction = nn.predict(input_data)
    print(f"Input: {input_data}, Predicted Output: {prediction}")
