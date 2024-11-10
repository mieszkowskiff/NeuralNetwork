import numpy as np
import csv



def read_data(file_name, classification=False):
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        data = np.array(list(csv_reader), dtype=np.float32)
    if classification:
        return data.T[0:2].T, data.T[2].T
    return data.T[0:1].T, data.T[1:2].T