import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    tab2=np.array([[1./2], [1], [2]])
    print(tab2)
    print(neural_network.softmax(tab2))
    print(sum(neural_network.softmax(tab2)))
    print(neural_network.softmax_derivative(tab2))
    
    X_train, Y_train = read_data.read_data("./data/regression/data.activation.train.1000.csv")
    X_test, Y_test = read_data.read_data(".data/regression/data.activation.test.1000.csv")
    
    # previous normalization [0,1]^2
    # min_x, max_x, min_y, max_y, X_train, Y_train = neural_network.data_normalization(X_train, Y_train)
    # centered normalization
    min_x, max_x, min_y, max_y, mean_x, mean_y, X_train, Y_train = neural_network.data_normalization2(X_train, Y_train)

    nn = neural_network.NeuralNetwork([1, 20, 20, 1], 12, 1, 40)
    nn.perform_training(X_train, Y_train, X_test, Y_test, min_x, max_x, min_y, max_y, mean_x, mean_y)
    #plt.plot(X_train, Y_train, "-g", label="Train")
    #plt.show()
    plt.plot(X_test, Y_test, "-b", label="Target")
    # previous normalization [0,1]^2
    # plt.plot(X_test, (nn(((X_test-min_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+min_y, "-r", label="Result")
    # centered normalization
    plt.plot(X_test, (nn(((X_test-mean_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+mean_y, "-r", label="Result")
    plt.legend(loc="upper left")
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.show()



    # comparison with wolfram alpha
    # test passed
    
    
    #plt.scatter(X_train, Y_train)
    #plt.plot(X_train, Y_train, "-g", label="Train")
    '''
    plt.scatter(X_test, Y_test)
    plt.scatter(X_test, (nn(((X_test-min_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+min_y)
    #plt.legend(loc="upper left")
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.show()
    '''
        