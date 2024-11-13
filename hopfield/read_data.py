import struct
import numpy as np


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in MNIST image file: {filename}')
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in MNIST label file: {filename}')
        
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels