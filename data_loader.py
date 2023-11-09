import csv
import numpy as np
import matplotlib.pyplot as plt

def read_data(start_index, filename = "./train.csv"):
    f = open(filename)
    lines = list( csv.reader(f) )
    for i in range(start_index, len(lines)):
        data = np.array(lines[i][1:], dtype = np.int64).reshape((28,28))
        answer = np.zeros((10,1))
        answer[int(lines[i][0])][0] = 1
        yield data, answer

    