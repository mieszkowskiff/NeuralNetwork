import numpy as np
import copy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_matrix_derivative(x):
    temp = sigmoid_derivative(x)
    m = np.zeros((len(temp), len(temp)))
    np.fill_diagonal(m, temp)
    return m

def leaky_ReLU(x):
    return np.maximum(x, 0.5*x)



def leaky_ReLU_derivative(x):
    def leaky_ReLU_derivative_lambda(x):
        if x<=0:
            return 0.5
        else:
            return 1
    return np.array(list(map(leaky_ReLU_derivative_lambda, x))).reshape(-1, 1)

def leaky_ReLU_matrix_derivative(x):
    temp = leaky_ReLU_derivative(x)
    m = np.zeros((len(temp), len(temp)))
    np.fill_diagonal(m, temp)
    return m

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-pow(tanh(x), 2)

def tanh_matrix_derivative(x):
    temp = tanh_derivative(x)
    m = np.zeros((len(temp), len(temp)))
    np.fill_diagonal(m, temp)
    return m

def softmax(x, T = 1):
    return np.exp(x/T) / np.sum(np.exp(x/T), axis=0)

def softmax_matrix_derivative(x):
    temp = softmax(x)
    m = -np.matmul(temp, temp.T)
    np.fill_diagonal(m, temp * (1 - temp))
    return m

def regression_data_normalization(x, l = -1, u = 1, x_min=None, x_max=None):
    if x_min is None:
        x_min = np.min(x, axis=1)
    if x_max is None:
        x_max = np.max(x, axis=1)
    x = (x - x_min) / (x_max - x_min)
    return x * (u - l) + l, x_min, x_max

def regression_data_denormalization(x, x_min, x_max, l = -1, u = 1):
    return (x - l) / (u - l) * (x_max - x_min) + x_min

def classification_data_normalization(x, mean=None, std=None):
    if mean is None:
        mean = np.array([np.mean(x, axis=1)]).T
    if std is None:
        std = np.array([np.std(x, axis=1)]).T
    return (x - mean) / std, mean, std

def classification_data_denormalization(x, mean, std):
    return x * std + mean

def one_hot_encoding(y):
    out = np.zeros((len(np.unique(y)), len(y)))
    for i in range(len(y)):
        out[int(y[i]), i] = 1
    return out

def one_hot_decoding(y):
    return np.argmax(y, axis=0)

def data_shuffle(x, y, classification=False):
    permute = np.random.permutation(x.shape[1])
    x = x[:,permute]
    if classification:
        y = y[permute]
    else:
        y = y[:,permute]
    return x, y

class NeuralNetwork:
    def __init__(self, structure, BATCH_SIZE=10, LEARNING_RATE=0.5, NUMBER_OF_EPOCHS=100, biases_present = True):
        self.structure = np.array(structure)
        self.layers = self.structure.shape[0] - 1

        self.biases_present = biases_present

        self.weights = [np.random.normal(0, 1/3, (self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        if self.biases_present:
            self.biases = [np.random.normal(0, 1/3, (self.structure[i + 1], 1)) for i in range(self.layers)]
        else:
            self.biases = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        #self.activation = sigmoid
        #self.activation_derivative = sigmoid_derivative
        self.activation = tanh
        self.activation_derivative = tanh_derivative
        self.last_layer_activation = tanh
        self.last_layer_activation_derivative = tanh_matrix_derivative

        self.neurons = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]
        self.chain = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.batch_size=BATCH_SIZE
        self.batch_size_counter = 0        #old batch_size variable, not used anywhere

        self.learning_rate=LEARNING_RATE

        self.number_of_epochs=NUMBER_OF_EPOCHS


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
    
    def cost(self, input, output):
        return np.sum((self.forward(input) - output)**2)


    def forward(self, input):
        #self.assure_input(input)
        self.neurons[0] = np.matmul(self.weights[0], input) + self.biases[0]
        for i in range(1, self.layers):
            # self.neurons, list of lists, containing forward propagation values of each neuron before
            # activation function is applied 
            self.neurons[i] = np.matmul(self.weights[i], self.activation(self.neurons[i - 1])) + self.biases[i]
        return self.last_layer_activation(self.neurons[-1])

    # variable output, from my understanding its NOT an output obtained from forward prop. for input.
    # These are labels, which, if Im correct, seem to have poor var. name   
    def calculate_chain(self, input, output):
        # this function prepares self.chain object which stores values of derivatives
        # with respect to particular neurons output (output before act. f.)
        self.forward(input)
        # SSE hardcoded here, cost function
        # still, math is correct in the line below
        self.chain[-1] = np.matmul(self.last_layer_activation_derivative(self.neurons[-1]),self.last_layer_activation(self.neurons[-1]) - output)
        for i in range(self.layers - 2, -1, -1):
            # math in this line also seems to be correct
            self.chain[i] = np.matmul(self.weights[i + 1].T, self.chain[i + 1]) * self.activation_derivative(self.neurons[i])
    
    def backward(self, input, output):
        self.calculate_chain(input, output)
        # weights_gradient/biases_gradient, objects storing the values of derivatives with respect to particular
        # NN parameters like weights or biases. Therefore weights_gradient is the same size as self.weights. Analogous
        # for biases. 
        self.weights_gradient[0] += np.matmul(self.chain[0], input.T)
        self.biases_gradient[0] += self.chain[0]
        for i in range(1, self.layers):
            # += operator is here for accumulating the back. prop. results from 
            # all instances appearing in the current batch 

            # math in the 2 following lines seems to be correct
            self.weights_gradient[i] += np.matmul(self.chain[i], self.activation(self.neurons[i - 1]).T)
            self.biases_gradient[i] += self.chain[i]
        
        self.batch_size_counter += 1

    def end_batch(self):
        for i in range(self.layers):
            # (dividing by the number of instances in one batch) (???)
            # /self.batch_size
            self.weights[i] -= (self.weights_gradient[i]/self.batch_size) * self.learning_rate
            if self.biases_present:
                self.biases[i] -= (self.biases_gradient[i]/self.batch_size) * self.learning_rate
        
        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]
        self.batch_size_counter = 0


    def perform_training(self, X_train, Y_train, X_test, Y_test):

        costs = []
        parameter_progress = []
        parameter_gradient_progress = []

        for j in range(self.number_of_epochs):
            if j%10 == 0:
                print("Epoch #", j)
                print(self.cost(X_test, Y_test))

    

            for i in range(X_train.shape[1]):
                self.backward(X_train[:,i:i+1], Y_train[:,i:i+1])

                if i % self.batch_size == self.batch_size - 1:
                    parameter_gradient_progress.append(self.weights_gradient[0][0][0])
                    self.end_batch()
            parameter_progress.append(self.weights[0][0][0])
            costs.append(self.cost(X_test, Y_test))
            

        return costs, parameter_progress, parameter_gradient_progress
    
    
        

    

    


                
                    




        
        
