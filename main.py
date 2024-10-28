import neural_network
import read_data
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    random.seed(10)

    # usual project data
    
    X_train, Y_train = read_data.read_data("C:/Users/Staś/Desktop/home/SN/data/reg/data.activation.train.1000.csv")
    X_test, Y_test = read_data.read_data("C:/Users/Staś/Desktop/home/SN/data/reg/data.activation.test.1000.csv")
    

    # generated dataset with noise
    '''
    X, Y = read_data.read_data("./data_gen_noise.txt")
    X, Y = neural_network.data_shuffle(X, Y)

    X_train, Y_train = X[0:700], Y[0:700]
    X_test, Y_test = X[700:1000], Y[700:1000]
    '''

    # glimpse at training and test data
    fig, ax = plt.subplots()
    x, y = X_train, Y_train
    ax.scatter(x, y, c='green', s=10, label='Training')
    
    x, y = X_test, Y_test
    ax.scatter(x, y, c='red', s=1, label='Test')
    
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.show()

    # centered normalization
    min_x, max_x, min_y, max_y, mean_x, mean_y, X_train, Y_train = neural_network.data_normalization(X_train, Y_train)

    # NN init, training, gathering info about the training process
    cost_epoch, weight_value, weight_error = [], [], []
    nn = neural_network.NeuralNetwork([1, 20, 20, 1], 1, 10, 0.1, 10)
    l, w, c = 2, 0, 8
    cost_epoch, weight_value, weight_error = nn.perform_training(X_train, Y_train, X_test, Y_test, min_x, max_x, min_y, max_y, mean_x, mean_y, l, w, c)
    
    # results
    fig, ax = plt.subplots()
    x, y = X_test, Y_test
    ax.scatter(x, y, c='green', s=10, label='Target')
    
    x, y = X_test, (nn(((X_test-mean_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+mean_y
    #x, y = X_test, (nn(((X_test-min_x)/(max_x-min_x)).reshape(-1, 1, 1)).reshape(-1))*(max_y-min_y)+min_y
    ax.scatter(x, y, c='red', s=1, label='Result')

    plt.axvline(x = min_x, color = 'b', label = 'Training dataset range')
    plt.axvline(x = max_x, color = 'b')
    
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.show()

    # Plotting 
    # history of cost value
    n=len(cost_epoch)
    fig, ax = plt.subplots()
    x, y = [i for i in range(n)], cost_epoch
    ax.scatter(x, y, c='green', s=30, label='Cost val. per epoch')
    
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()
    
    # weight value
    n=len(weight_value)
    fig, ax = plt.subplots()
    x, y = [i for i in range(n)], weight_value
    ax.scatter(x, y, c='blue', s=1, label='weight value')
    
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()

    # alternative
    '''
    plt.plot(x, y)
    plt.show()
    '''
    
    # corresponding weight error
    n=len(weight_error)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    x, y = [i for i in range(n)], weight_error

    ax1.scatter(x, y, c='blue', s=1, label='weight error')
    ax2.scatter(x, y, c='blue', s=1, label='weight error')

    ax2.set_yscale("log")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax1.grid(True)
    ax2.grid(True)
    plt.show()