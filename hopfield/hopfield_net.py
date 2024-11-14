import numpy as np
import display

def heavy_side(x):
    return np.heaviside(x, 0)

def signum(x):
    return 2*heavy_side(x)-1

class HopfieldNet:
    def __init__(self, lm, n, activation, dynamics):

        self.lm = lm
        self.n = n
        self.dynamics=dynamics

        self.W = np.zeros((n, n), dtype=np.float32)

        if activation == 'signum':
            self.activation = signum
        elif activation == 'heaviside':
            self.activation = heavy_side

    def training(self, X):
        if self.lm=='HEBB':
            for x in X:
                self.W += np.outer(x, x)
            np.fill_diagonal(self.W, 0)
            self.W /= self.n
        elif self.lm=='OJA':
            print("Not yet :)")

    def call(self, x):
        if self.dynamics=='asynchronous':
            for i in range(self.n):
                u = np.dot(self.W[i], x)
                x[i] = self.activation(u)
        elif self.dynamics=='synchronous':
            u = np.dot(self.W, x)
            x = self.activation(u)
        return x

    def forward(self, dims, init_x, epochs, show_vis=0):
        x = np.array(init_x)
        for j in range(epochs):
            x = self.call(x)
            if ( show_vis!=0 and (j % show_vis == show_vis - 1) ):
                display.display(x, dims)
        return x