import numpy as np
import copy
import statistics as stat

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) - pow(sigmoid(x), 2)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-pow(tanh(x), 2)

def leaky_ReLU(x):
    return np.maximum(x, 0.5*x)

def leaky_ReLU_derivative_lambda(x):
    if x<=0:
        return 0.5
    else:
        return 1

def leaky_ReLU_derivative(x):
    return np.array(list(map(leaky_ReLU_derivative_lambda, x))).reshape(-1, 1)

def identity(x):
    return x

def identity_derivative(x):
    return 1

def SSE(x, y):
    return sum(pow(x-y,2))

def SSE_derivative(x, y):
    return 1

def data_normalization(x, y):
    min_x=np.min(x)
    max_x=np.max(x)
    min_y=np.min(y)
    max_y=np.max(y)

    mean_x=stat.mean(x)
    mean_y=stat.mean(y)

    x = (x - mean_x) / (max_x - min_x)
    #x = (x - min_x) / (max_x - min_x)
    y = (y - mean_y) / (max_y - min_y)
    #y = (y - min_y) / (max_y - min_y)
    print(np.min(x))
    print(np.max(x))
    print(np.min(y))
    print(np.max(y))
    return min_x, max_x, min_y, max_y, mean_x, mean_y, x, y

def data_shuffle(x, y):
    permute = np.random.permutation(len(x))
    x = x[permute]
    y = y[permute]
    return x, y

class NeuralNetwork:
    def __init__(self, structure, bias_presence=1, BATCH_SIZE=10, LEARNING_RATE=0.5, NUMBER_OF_EPOCHS=20):
        self.structure = np.array(structure)
        self.layers = self.structure.shape[0] - 1

        self.bias_presence = bias_presence

        self.weights = [np.random.uniform(-1, 1, (self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases = [np.random.uniform(-1, 1, (self.structure[i + 1], 1)) for i in range(self.layers)]    
        # if bias_presence=0, initial values for biases are set to 0
        # bias_presence in end_batch() function assures these values will never update 
        if self.bias_presence==0:
            n=len(self.biases)
            self.biases=[self.biases[i]*0 for i in range(n)]

        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

        self.last_layer_activation = tanh
        self.last_layer_activation_derivative = tanh_derivative

        self.neurons = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]
        self.chain = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

        self.batch_size=BATCH_SIZE

        self.learning_rate=LEARNING_RATE

        self.number_of_epochs=NUMBER_OF_EPOCHS


    def assure_input(self, input):
        assert type(input) is np.ndarray
        assert input.shape[0] == self.structure[0]


    def assure_output(self, output):
        assert type(output) is np.ndarray
        assert output.shape[0] == self.structure[-1]


    def __call__(self, input):
        data = copy.deepcopy(input)
        for i in range(self.layers):
            data = self.activation(np.matmul(self.weights[i], data) + self.biases[i])
        return data


    def forward(self, input):
        self.neurons[0] = np.matmul(self.weights[0], input) + self.biases[0]
        for i in range(1, self.layers):
            # self.neurons, list of lists, containing forward propagation values of each neuron before
            # activation function is applied 
            self.neurons[i] = np.matmul(self.weights[i], self.activation(self.neurons[i - 1])) + self.biases[i]
        return self.last_layer_activation(self.neurons[-1])

    # variable output, labels/target values   
    def calculate_chain(self, input, output):
        # this function prepares self.chain object which stores values of derivatives
        # with respect to particular neurons output (output before act. f.)
        self.forward(input)
        self.assure_output(output)
        # SSE hardcoded here, cost function
        # W przypadku klasyfikacji ta linijka to: macierz różniczki f. akt. * kolumna różniczki f. kosztu
        #self.chain[-1] = np.matmul(self.last_layer_activation_derivative(self.neurons[-1]), self.last_layer_activation(self.neurons[-1]) - output)
        self.chain[-1] = (self.last_layer_activation(self.neurons[-1]) - output) * self.last_layer_activation_derivative(self.neurons[-1])
        #self.chain[-1] = (self.activation(self.neurons[-1]) - output) * self.activation_derivative(self.neurons[-1])
        for i in range(self.layers - 2, -1, -1):
            self.chain[i] = np.matmul(self.weights[i + 1].T, self.chain[i + 1]) * self.activation_derivative(self.neurons[i])
    
    def backward(self, input, output):
        self.assure_input(input)
        self.assure_output(output)

        self.calculate_chain(input, output)
        # weights_gradient/biases_gradient, objects storing the values of derivatives with respect to particular
        # NN parameters like weights or biases. Therefore weights_gradient is the same size as self.weights. Analogous
        # for biases. 
        self.weights_gradient[0] += np.matmul(self.chain[0], input.T)
        self.biases_gradient[0] += self.chain[0]
        for i in range(1, self.layers):
            # += operator is here for accumulating the back. prop. results from 
            # all instances appearing in the current batch 
            self.weights_gradient[i] += np.matmul(self.chain[i], self.activation(self.neurons[i - 1]).T)
            self.biases_gradient[i] += self.chain[i]
        

    def end_batch(self):
        for i in range(self.layers):
            # dividing by the number of instances in one batch, normalization /self.batch_size
            self.weights[i] -= (self.weights_gradient[i]/self.batch_size) * self.learning_rate
            # NN without biases on layers, so with bias_presence=0, will never have its biases updated 
            if self.bias_presence==1:
                self.biases[i] -= (self.biases_gradient[i]/self.batch_size) * self.learning_rate
        
        self.weights_gradient = [np.zeros((self.structure[i + 1], self.structure[i])) for i in range(self.layers)]
        self.biases_gradient = [np.zeros((self.structure[i + 1], 1)) for i in range(self.layers)]

    def perform_training(self, X_train, Y_train, X_test, Y_test, min_x, max_x, min_y, max_y, mean_x, mean_y, l, w, c):
        flag=False
        #check if user is a nice person
        if (w==int(w) and c==int(c) and l==int(l)):
            if (0<=l<=self.layers-1):
                max_w=self.weights[l].shape[0]
                max_c=self.weights[l].shape[1]
                if (0<=w<=max_w-1) and (0<=c<=max_c-1):
                        flag=True

        X_len=len(X_train)
        cost_epoch=[]
        weight_error=[]
        weight_value=[]
        for j in range(self.number_of_epochs):
            if j%2==0 :
                # data shuffle for every 2nd epoch to prevent feeding exactly the same batches
                # in every epoch
                X_train, Y_train = data_shuffle(X_train, Y_train)    
                print("Epoch #", j)
                # data_normalization
                print(SSE((self(((X_test-mean_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+mean_y, Y_test))
                #print(SSE((self(((X_test-min_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+min_y, Y_test))
            
            cost_epoch.append(SSE((self(((X_test-mean_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+mean_y, Y_test))
            #cost_epoch.append(SSE((self(((X_test-min_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+min_y, Y_test))

            for i in range(X_len):
                self.backward(np.array([[X_train[i]]]), np.array([[Y_train[i]]]))
                # last batch, if X_len is not divisible by batch_size, is not used
                if (i % self.batch_size == 0):
                    if flag:
                        weight_value.append(self.weights[l][w][c])
                        weight_error.append(self.weights_gradient[l][w][c]/self.batch_size)
                    self.end_batch()
        return cost_epoch, weight_value, weight_error

    

    


                
                    




        
        
