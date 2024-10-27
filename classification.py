import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    np.random.seed(1)
    X_train, Y_train = read_data.read_data("./data/classification/data.simple.train.1000.csv", True)
    X_test, Y_test = read_data.read_data("./data/classification/data.simple.test.1000.csv", True)

    X_train = X_train.T
    Y_train = Y_train - 1
    X_test = X_test.T
    Y_test = Y_test - 1

    X_train, mean, std = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean, std)
    Y_train = neural_network.one_hot_encoding(Y_train)

    nn = neural_network.NeuralNetwork([2, 2], 5, 0.1, 150)
    

    Y_pred = nn.forward(X_test)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    out = Y_pred + 2 * Y_test
    plt.scatter(X_test[0], X_test[1], c=out)
    plt.show()

    costs, weights = nn.perform_classification_training(X_train, Y_train, X_test, Y_test)

    plt.plot(costs)
    plt.show()

    plt.plot(weights)
    plt.show()



    Y_pred = nn.forward(X_test)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    out = Y_pred + 2 * Y_test

    plt.scatter(X_test[0], X_test[1], c=out)
    plt.show()


    
    



    
    

    
