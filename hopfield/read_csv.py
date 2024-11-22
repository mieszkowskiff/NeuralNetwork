import re
import csv
import os
import numpy as np

CONST_DIMS_PATTERN = r'-\d+x\d+'
CONST_NUM_OF_DIMS = 2

def find_dimensions_in_file_path(text):
    dim = re.findall(CONST_DIMS_PATTERN, text)
    dim = str(dim[-1])
    dim = str(dim).replace("-", "")
    dim = dim.split("x")
    dim = [int(dim[i]) for i in range(CONST_NUM_OF_DIMS)]
    return dim

def read_patterns(csv_file_path):
    
    dims = find_dimensions_in_file_path(csv_file_path)

    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file, delimiter=',')

        rows=[]
        for row in csv_reader:
            rows.append(np.array([int(x) for x in row]))
    return np.array(rows), dims

def noise(example, noise_level = 0.01):
    """
    Add noise to an example.
    """

    # Copy the example to avoid modifying the original
    example = np.copy(example)

    # Generate random indices
    indices = np.random.choice(range(len(example)), int(noise_level * len(example)), replace=False)

    # Flip the bits at the random indices
    example[indices] = -example[indices]

    return example