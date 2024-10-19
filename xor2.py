import numpy as np

# Funkcja aktywacji sigmoid, zamienia wejścia na wartości w przedziale (0, 1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji sigmoid, niezbędna do backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)  # x w tym miejscu to już wyjście funkcji sigmoid

# Klasa Sieci Neuronowej
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicjalizacja wag losowymi wartościami dla warstwy wejściowej -> ukrytej
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        
        # Inicjalizacja wag losowymi wartościami dla warstwy ukrytej -> wyjściowej
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        
        # Biasy (przesunięcia) dla warstwy ukrytej i wyjściowej
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    # Propagacja w przód: liczymy wyniki wyjściowe dla danych wejściowych
    def feedforward(self, X):
        # Obliczanie wartości neuronów w warstwie ukrytej (wartości przed aktywacją)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        
        # Obliczanie wyjść neuronów w warstwie ukrytej po aktywacji sigmoid
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Obliczanie wartości neuronów w warstwie wyjściowej (wartości przed aktywacją)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        
        # Obliczanie wyjść warstwy wyjściowej (po aktywacji sigmoid)
        self.final_output = sigmoid(self.final_input)
        
        # Zwracamy wynik wyjściowy sieci
        return self.final_output

    # Propagacja wstecz: obliczanie pochodnych i aktualizacja wag
    def backpropagation(self, X, y, output, learning_rate):
        # Krok 1: Obliczenie błędu wyjścia (różnica między przewidywanym a rzeczywistym wyjściem)
        error = y - output  # y to rzeczywista wartość, output to przewidywane wyjście
        
        # Krok 2: Pochodna funkcji kosztu względem wyjścia sieci
        d_output = error * sigmoid_derivative(output)  # Gradient błędu dla warstwy wyjściowej
        
        # Krok 3: Propagowanie błędu wstecz do warstwy ukrytej
        # Błąd dla warstwy ukrytej to pochodna błędu w warstwie wyjściowej względem wyjść ukrytych
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        
        # Pochodna funkcji kosztu względem wyjść z warstwy ukrytej
        d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)  # Gradient błędu dla warstwy ukrytej
        
        # Krok 4: Aktualizacja wag i biasów
        # Aktualizacja wag dla połączeń warstwa ukryta -> wyjściowa
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        
        # Aktualizacja wag dla połączeń warstwa wejściowa -> ukryta
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        
        # Aktualizacja biasów
        self.bias_output += np.sum(d_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    # Funkcja trenująca sieć przez wiele epok
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Propagacja w przód
            output = self.feedforward(X)
            
            # Propagacja wstecz
            self.backpropagation(X, y, output, learning_rate)
            
            # Co 1000 epok wyświetlamy aktualny stan błędu
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))  # MSE
                print(f'Epoch {epoch}, Loss: {loss:.5f}')

    # Funkcja do przewidywania wyników dla nowych danych
    def predict(self, X):
        return self.feedforward(X)

# Zbiór danych dla bramki XOR (logiczne wejścia i oczekiwane wyjścia)
X = np.array([[0, 0],  # Wejście 0 XOR 0 -> Oczekiwane wyjście: 0
              [0, 1],  # Wejście 0 XOR 1 -> Oczekiwane wyjście: 1
              [1, 0],  # Wejście 1 XOR 0 -> Oczekiwane wyjście: 1
              [1, 1]]) # Wejście 1 XOR 1 -> Oczekiwane wyjście: 0

y = np.array([[0],  # Oczekiwane wyjście dla [0, 0]
              [1],  # Oczekiwane wyjście dla [0, 1]
              [1],  # Oczekiwane wyjście dla [1, 0]
              [0]]) # Oczekiwane wyjście dla [1, 1]



np.random.seed(0)
# Inicjalizacja sieci neuronowej
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

print(nn.predict(X))
# Trenowanie sieci neuronowej
nn.train(X, y, epochs=5000, learning_rate=0.1)

print(nn.predict(X))

