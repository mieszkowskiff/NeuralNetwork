from neural_network import NeuralNetwork
import numpy as np
import data_loader
import pickle

def main():
    AI = NeuralNetwork([784, 16, 16, 10])
    batch_size = 8
    file = open("state", "rb")
    data = pickle.load(file)
    file.close()
    AI.weights = data["weights"]
    AI.biases = data["biases"]
    index = data["index"]
    try:
        for data in data_loader.read_data(index):
            digit, answer = data
            AI.calculate_gradient(digit.reshape((784,1)), answer)
            index += 1
            if index % batch_size == 0:
                print(index)
                AI.end_batch(0.01)
        file = open('state', 'wb')
        pickle.dump(
                    {
                        "weights": AI.weights,
                        "biases": AI.biases,
                        "index": 1
                    },
                    file
                )
        file.close()


    except KeyboardInterrupt:
        for data in data_loader.read_data(index):
            digit, answer = data
            AI.calculate_gradient(digit.reshape((784,1)), answer)
            index += 1
            if index % batch_size == 0:
                AI.end_batch(0.01)
                file = open('state', 'wb')
                pickle.dump(
                    {
                        "weights": AI.weights,
                        "biases": AI.biases,
                        "index": index
                    },
                    file
                )
                file.close()







if __name__ == "__main__":
    main()