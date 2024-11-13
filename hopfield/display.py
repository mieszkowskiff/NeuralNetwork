import matplotlib.pyplot as plt



def display(x, width, height):
    plt.imshow(x.reshape((width, height)), cmap='gray')
    plt.show()