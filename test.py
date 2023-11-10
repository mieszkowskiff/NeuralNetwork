from neural_network import NeuralNetwork
import numpy as np
import data_loader
import pickle


AI = NeuralNetwork([784, 128, 128, 10])
file = open("state", "rb")
data = pickle.load(file)
file.close()
AI.weights = data["weights"]
AI.biases = data["biases"]

cost = 0
for data in data_loader.read_data(1):
    digit, answer = data
    cost += AI.calculate_cost(digit.reshape((784, 1)), answer)

print(cost)
#37859.550295630754