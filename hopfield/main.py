import display
import hopfield_net
import read_data

if __name__ == '__main__':
    data = read_data.load_mnist_images('./hopfield/data/mnist/train-images.idx3-ubyte')
    labels = read_data.load_mnist_labels('./hopfield/data/mnist/train-labels.idx1-ubyte')
    print(labels)


   