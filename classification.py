import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt
from ploting import plot_classification



if __name__ == "__main__":
    np.random.seed(1)
    X_train, Y_train = read_data.read_data("./data/classification/data.three_gauss.train.1000.csv", True)
    X_test, Y_test = read_data.read_data("./data/classification/data.three_gauss.test.1000.csv", True)

    X_train = X_train.T
    Y_train = Y_train - 1
    X_test = X_test.T
    Y_test = Y_test - 1

    X_train, Y_train = neural_network.data_shuffle(X_train, Y_train, True)
    X_train, mean, std = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean, std)
    Y_train = neural_network.one_hot_encoding(Y_train)

    n_classes = Y_train.shape[0]
    nn = neural_network.NeuralNetwork([2, 10, 10, 10, n_classes], 5, 0.1, 20)
    

    plot_classification(X_test, Y_test, n_classes)
    
    costs, parameter, parameter_gradient = nn.perform_training(X_train, Y_train, X_test, Y_test)

    Y_pred = nn.forward(X_test)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    out = Y_pred + n_classes * Y_test

    plot_classification(X_test, out, n_classes * n_classes)


    
    



    
    

    
