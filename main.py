import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x, y = read_data.read_data("/home/filip/Desktop/NeuralNetwork/data/regression/data.activation.test.100.csv")
    permute = np.random.permutation(len(x))
    x = x[permute]
    y = y[permute]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    plt.scatter(x, y)
    nn = neural_network.NeuralNetwork([1, 2, 1])
    for j in range(1000):
        for i in range(100):
            nn.backward(np.array([[x[i]]]), np.array([[y[i]]]))
            if i % 10 == 0:
                nn.end_batch(0.1)
    plt.scatter(x, nn(x.reshape(-1, 1, 1)))
    plt.savefig("plot.png")
    
        