import matplotlib.pyplot as plt
import numpy as np
import csv


# Replace 'your_file.csv' with the path to your actual CSV file
with open('./data/regression/data.activation.test.500.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    csv_reader.__next__()
    data = np.array(list(csv_reader), dtype=np.float32)
x = data.T[0].T
y = data.T[1].T

plt.scatter(x, y)
plt.savefig('plot.png')
        