from neural_network import NeuralNetwork
import pickle

def main():
    emptyAI = NeuralNetwork([784, 16, 16, 10])
    file = open('state', 'wb')
    pickle.dump(
        {
            "weights": emptyAI.weights,
            "biases": emptyAI.biases,
            "index": 0
        },
        file
    )
    file.close()




if __name__ == "__main__":
    main()