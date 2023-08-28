from neural_network import NeuralNetwork
import numpy as np
import data_loader

def main():

    
    AI = NeuralNetwork([784, 16, 16, 10])
    for d in data_loader.read_data(42000):






if __name__ == "__main__":
    main()