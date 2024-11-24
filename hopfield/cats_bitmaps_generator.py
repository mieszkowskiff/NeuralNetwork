import display
import matplotlib.pyplot as plt
import numpy as np

path = "./data/hopfield/photos/"
shape = (200, 200)
cats_number = 2

photos = []


for i in range(cats_number):
    file_name = path + "cat" + str(i) + ".png"
    bitmap = np.array(display.convert_to_bitmap(file_name, shape))
    photos.append(bitmap.reshape(-1))

photos = np.array(photos)

np.savetxt(path + "../cats-" + str(shape[0]) + "x" + str(shape[1]) + ".csv", photos, delimiter=",", fmt="%d")


