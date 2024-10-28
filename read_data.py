import numpy as np
import csv

def read_data(file_name):
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        data = np.array(list(csv_reader), dtype=np.float32)
    return data.T[0].T, data.T[1].T