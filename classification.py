import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    np.random.seed(12)
    X_train, Y_train = read_data.read_data("./data/classification/data.simple.train.1000.csv", True)
    X_test, Y_test = read_data.read_data("./data/classification/data.simple.test.1000.csv", True)

    X_train = X_train.T
    Y_train = Y_train - 1
    X_test = X_test.T
    Y_test = Y_test - 1

    X_train, mean, std = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean, std)
    Y_train = neural_network.one_hot_encoding(Y_train)

    nn = neural_network.NeuralNetwork([2,2, 2], 5, 0.5, 15000)
    nn.perform_classification_training(X_train, Y_train)

    Y_pred = np.array([nn.forward(X_test[:,i:i+1]) for i in range(X_test.shape[1])]).T.reshape(2, -1)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    out = Y_pred + 2 * Y_test
    print(out)
    plt.scatter(X_test[0], X_test[1], c=out)
    plt.savefig("classification.png")


    
    



    
    

    
