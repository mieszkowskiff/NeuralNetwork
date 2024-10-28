import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(56)
    X_train, Y_train = read_data.read_data("./data/regression/data.activation.train.1000.csv", False)
    X_test, Y_test = read_data.read_data("./data/regression/data.activation.test.1000.csv", False)
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T

    fig, ax = plt.subplots()
    ax.scatter(X_train, Y_train, c='b', label='Training data')
    ax.scatter(X_test, Y_test, c='r', label='Testing data')
    plt.legend()
    plt.title("Training and testing data")
    plt.show()


    
    Y_train, min_y, max_y = neural_network.regression_data_normalization(Y_train)
    Y_test, _, _ = neural_network.regression_data_normalization(Y_test, x_min = min_y, x_max = max_y)

    X_train, mean_x, std_x = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean_x, std_x)
    
    X_train, Y_train = neural_network.data_shuffle(X_train, Y_train)

    nn = neural_network.NeuralNetwork([1, 3, 1], 5, 0.1, 150)

    cost, parameter_progress, parameter_gradient_progress = nn.perform_training(X_train, Y_train, X_test, Y_test)

    Y_pred = nn.forward(X_test)

    Y_pred = neural_network.regression_data_denormalization(Y_pred, min_y, max_y)
    Y_test = neural_network.regression_data_denormalization(Y_test, min_y, max_y)
    X_test = neural_network.classification_data_denormalization(X_test, mean_x, std_x)

    fig, ax = plt.subplots()
    ax.scatter(range(len(cost) - 1), cost[1:], c='b', label='Cost')
    plt.legend()
    plt.title("Cost function over epochs")
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(range(len(parameter_progress)), parameter_progress, c='b', label='Chosen parameter')
    plt.legend()
    plt.title("Value of chosen parameter after each batch")
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(range(len(parameter_gradient_progress)), parameter_gradient_progress, c='b', label='Chosen parameter gradient')
    plt.legend()
    plt.title("Value of chosen parameter gradient after each batch")
    plt.show()



    fig, ax = plt.subplots()
    ax.scatter(X_test, Y_pred, c='b', label='Prediction')
    ax.scatter(X_test, Y_test, c='r', label='True value')
    plt.title("Prediction vs True value")
    plt.legend()
    plt.show()


    
        