import numpy as np





class HopfieldNet:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n), dtype=np.float32)

    def hebb_training(self, X):
        for x in X:
            self.W += np.outer(x, x)
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def asynchronous_call(self, x):
        for i in range(1000):
            i = np.random.randint(0, self.n)
            u = np.dot(self.W[i], x)
            x[i] = np.sign(u)
        return x
    
    def synchronous_call(self, x):
        for i in range(1000):
            u = np.dot(self.W, x)
            x = np.sign(u)
        return x

    
