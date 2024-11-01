import read_data
import ai
import numpy as np

def mnist_normalize(X):
    return X / 255




if __name__ == "__main__":
    np.random.seed(0)
    print("Loading data...")
    X_train = read_data.load_mnist_images('./data/mnist/train-images.idx3-ubyte')
    Y_train = read_data.load_mnist_labels('./data/mnist/train-labels.idx1-ubyte')
    
    X_test = read_data.load_mnist_images('./data/mnist/t10k-images.idx3-ubyte')
    Y_test = read_data.load_mnist_labels('./data/mnist/t10k-labels.idx1-ubyte')

    print("Data loaded.")
    print("Processing data...")
    num_train_images = X_train.shape[0]
    num_test_images = X_test.shape[0]


    Y_train = ai.one_hot_encoding(Y_train)
    Y_test = ai.one_hot_encoding(Y_test)

    X_train = X_train.reshape(num_train_images, -1).T
    X_train = mnist_normalize(X_train)

    X_test = X_test.reshape(num_test_images, -1).T
    X_test = mnist_normalize(X_test)

    print("Data processed.")
    print("Shuffling data...")


    X_train, Y_train = ai.data_shuffle(X_train, Y_train)

    print("Data shuffled.")
    print("Creating neural network...")
    
    hidden_layer_size = 130

    nn = ai.NeuralNetwork(
        [
            X_train.shape[0], 
            hidden_layer_size, 
            hidden_layer_size, 
            Y_train.shape[0]
        ], 
        BATCH_SIZE = 30, 
        LEARNING_RATE = 0.1, 
        NUMBER_OF_EPOCHS = 7, 
        activation = 'sigmoid', 
        last_layer_activation = 'softmax'
    )
    print("Neural network created.")

    costs, parameter_progress, parameter_gradient_progress = nn.perform_training(X_train, Y_train, X_test, Y_test)
    print(nn.calculate_accuracy(X_test, Y_test))
    np.savez(f'./models/weights-{hidden_layer_size}.npz', *nn.weights)
    np.savez(f'./models/biases-{hidden_layer_size}-.npz', *nn.biases)

    