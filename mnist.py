import read_data
import ai

def mnist_normalize(X):
    return X / 255




if __name__ == "__main__":
    X_train = read_data.load_mnist_images('./data/mnist/train-images.idx3-ubyte')
    Y_train = read_data.load_mnist_labels('./data/mnist/train-labels.idx1-ubyte')
    num_images = X_train.shape[0]

    Y_trian = ai.one_hot_encoding(Y_train)


    X_train = X_train.reshape(num_images, -1).T
    X_train = mnist_normalize(X_train)

    X_train, Y_train = ai.data_shuffle(X_train, Y_train, classification=True)
    
    

    