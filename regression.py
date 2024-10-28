import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X_train, Y_train = read_data.read_data("./data/regression/data.cube.train.1000.csv", False)
    X_test, Y_test = read_data.read_data("./data/regression/data.cube.test.1000.csv", False)
    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T

    
    Y_train, min_y, max_y = neural_network.regression_data_normalization(Y_train)
    Y_test, _, _ = neural_network.regression_data_normalization(Y_test, x_min = min_y, x_max = max_y)

    X_train, mean_x, std_x = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean_x, std_x)
    
    X_train, Y_train = neural_network.data_shuffle(X_train, Y_train)

    nn = neural_network.NeuralNetwork([1, 3, 1], 5, 0.5, 150)

    nn.perform_training(X_train, Y_train)

    Y_pred = nn.forward(X_test)
    
    plt.scatter(X_test, Y_pred, c='b')
    plt.scatter(X_test, Y_test, c='r')
    plt.show()


    
        