from neural_network import NeuralNetwork
import numpy as np
import data_loader
import pickle

def main():
    emptyAI = NeuralNetwork([784, 16, 16, 10])
    trainedAI = NeuralNetwork([784, 16, 16, 10])
    file = open("state", "rb")
    data = pickle.load(file)
    file.close()
    trainedAI.weights = data["weights"]
    trainedAI.biases = data["biases"]

    cost_emptyAI = 0
    cost_trainedAI = 0
    for d in data_loader.read_data(1):
        data, answer = d
        cost_emptyAI += emptyAI.calculate_cost(data.reshape((784, 1)), answer)
        cost_trainedAI += trainedAI.calculate_cost(data.reshape((784, 1)), answer)
    print(cost_emptyAI)
    print(cost_trainedAI)

if __name__ == "__main__":
    main()