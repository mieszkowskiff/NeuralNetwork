from neural_network import NeuralNetwork
import pickle

def main():
    emptyAI = NeuralNetwork([784, 128, 128, 10])
    file = open('state', 'wb')
    pickle.dump(
        {
            "weights": emptyAI.weights,
            "biases": emptyAI.biases,
            "index": 1
        },
        file
    )
    file.close()




if __name__ == "__main__":
    main()