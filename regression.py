import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(56)
    X_train, Y_train = read_data.read_data("./data/projekt1/regression/data.cube.train.1000.csv", False)
    X_test, Y_test = read_data.read_data("./data/projekt1/regression/data.cube.test.1000.csv", False)
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T

    fig, ax = plt.subplots()
    ax.scatter(X_train, Y_train, s=10, c='b', label='Training data')
    ax.scatter(X_test, Y_test, s=1, c='r', label='Testing data')
    plt.legend()
    plt.title("Training and testing data")
    ax.grid(True)
    plt.show()
    
    Y_train, min_y, max_y = neural_network.regression_data_normalization(Y_train)
    Y_test, _, _ = neural_network.regression_data_normalization(Y_test, x_min = min_y, x_max = max_y)

    X_train, mean_x, std_x = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean_x, std_x)
    
    X_train, Y_train = neural_network.data_shuffle(X_train, Y_train)

    nn = neural_network.NeuralNetwork([1, 10, 1], 5, 0.5, 50)

    cost, parameter_progress, parameter_gradient_progress = nn.perform_training(X_train, Y_train, X_test, Y_test)


    if False:
        fig, ax = plt.subplots()
        ax.scatter([i for i in range(len(cost))], cost, c='b', s=10,  label='Cost')
        plt.title("Cost over epochs")
        plt.legend()
        ax.grid(True)
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter([i for i in range(len(parameter_progress))], parameter_progress, c='b', s=10,  label='parameter')
        plt.title("Chosen parameter over epochs")
        plt.legend()
        ax.grid(True)
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter([i for i in range(len(parameter_gradient_progress))], parameter_gradient_progress, c='b', s=10,  label='parameter gradient')
        plt.title("Chosen parameter gradient over epochs")
        plt.legend()
        ax.grid(True)
        plt.show()

    

    Y_pred = nn.forward(X_test)

    Y_pred = neural_network.regression_data_denormalization(Y_pred, min_y, max_y)
    Y_test = neural_network.regression_data_denormalization(Y_test, min_y, max_y)
    X_test = neural_network.classification_data_denormalization(X_test, mean_x, std_x)

    fig, ax = plt.subplots()
    ax.scatter(X_test, Y_pred, c='b', s=1,  label='Prediction')
    ax.scatter(X_test, Y_test, c='r', s=1, label='True value')
    plt.title("Prediction vs True value")
    plt.legend()
    ax.grid(True)
    plt.show()


    
        