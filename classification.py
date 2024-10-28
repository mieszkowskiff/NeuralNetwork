import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("TkAgg")

def plot(X, out, classes = 2):
    if classes == 2:
        colors = ['blue', 'darkred', 'red', 'mediumblue']
    if classes == 3:
        colors = ['blue', 'darkred', 'red', 'firebrick', 'mediumblue', 'red', 'darkred', 'firebrick', 'royalblue']
    cmap = mcolors.ListedColormap(colors)
    plt.scatter(X[0], X[1], c=out, cmap=cmap)
    legend_labels = np.unique(out)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{int(label)}', 
                        markerfacecolor=color, markersize=10) 
            for label, color in zip(legend_labels, colors)]
    plt.legend(handles=handles, title='Out Values', loc='upper right')
    plt.show()




if __name__ == "__main__":
    np.random.seed(1)
    X_train, Y_train = read_data.read_data("./data/classification/data.three_gauss.train.1000.csv", True)
    X_test, Y_test = read_data.read_data("./data/classification/data.three_gauss.test.1000.csv", True)

    X_train = X_train.T
    Y_train = Y_train - 1
    X_test = X_test.T
    Y_test = Y_test - 1

    X_train, mean, std = neural_network.classification_data_normalization(X_train)
    X_test, _, _ = neural_network.classification_data_normalization(X_test, mean, std)
    Y_train = neural_network.one_hot_encoding(Y_train)

    nn = neural_network.NeuralNetwork([2, 10, 3], 5, 0.1, 20)
    

    Y_pred = nn.forward(X_test)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    out = Y_pred + 3 * Y_test

    plot(X_test, out, 3)
    

    costs, weights, weights_gradient = nn.perform_training(X_train, Y_train, X_test, Y_test)

    plt.plot(costs)
    plt.show()

    plt.plot(weights)
    plt.show()



    Y_pred = nn.forward(X_test)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    out = Y_pred + 3 * Y_test

    plot(X_test, out, 3)


    
    



    
    

    
