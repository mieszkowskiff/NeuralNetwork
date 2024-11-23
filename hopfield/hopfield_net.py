import numpy as np
import matplotlib.pyplot as plt
import display
from matplotlib.animation import FuncAnimation
import copy
import math

def heavy_side(x):
    return np.heaviside(x, 0)

def signum(x):
    return 2*heavy_side(x)-1

class HopfieldNet:
    def __init__(self, n, activation, dynamics):
        self.n = n
        self.dynamics=dynamics

        self.W = np.zeros((n, n), dtype=np.float32)
        self.b = np.zeros(n, dtype=np.float32)

        if activation == 'signum':
            self.activation = signum
        elif activation == 'heaviside':
            self.activation = heavy_side

    def HEBB_training(self, X):
        for x in X:
            self.W += np.outer(x, x)
            self.b += x
        np.fill_diagonal(self.W, 0)
        # self.W /= self.n
        self.W /= X.shape[0]
        self.b /= X.shape[0]
        self.b *= 1

    # eta usually from range [0.7, 0.9] (???)
    def OJA_training(self, X, training_epochs = 10, eta = 0.01):
        self.HEBB_training(X)
        E = training_epochs
        M = X.shape[0]
        N = X.shape[1]
        print(M)
        X=X.T
        
        for k in range(E):
            #W_copy=copy.deepcopy(self.W)
            #print(W_copy[20])
            print("Oja epoch #", k+1)
            print("row: ", self.W[5][0:6])
            for i in range(self.n):
                V=np.matmul(self.W, X)      #w_{ij} ksi_j
                Dv=np.zeros(self.n, dtype=np.float32)
                '''
                print(V.shape)
                print(X.shape)
                print(self.W.shape)
                print(V[:,0].shape)
                print(X[:,0].shape)
                print(self.W[:, i].shape)
                '''
                for s in range(M):
                    Dv += V[:,s] * (X[:,s] - self.W[i] * V[:,s])
                Dv[i] = 0
                #print(Dv[0:6])
                self.W[i] += ( eta/M ) * Dv

    def call(self, x):
        if self.dynamics=='asynchronous':
            for i in range(self.n):
                u = np.dot(self.W[i], x) + self.b[i]
                #try:
                x[i] = self.activation(u)
                #except:
                    #print("### ERROR ###")
                    #print(u)
        elif self.dynamics=='synchronous':
            u = np.dot(self.W, x) + self.b
            x = self.activation(u)
        return x

    def forward(self, dims, init_x, epochs, animation = False):
        x = np.array(init_x)
        if animation:
            frames = [copy.deepcopy(x)]
        #display.display(x, dims)
        for j in range(epochs):
            x = self.call(x)
            if animation:
                frames.append(copy.deepcopy(x))
                
        if animation:
            fig, ax = plt.subplots()
            image = ax.imshow(frames[0].reshape(dims[1], dims[0]), cmap='gray', vmin=0, vmax=1)
            def update(frame):
                image.set_data(frames[frame].reshape(dims[1], dims[0]))
                return [image]
            anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
            anim.save('animation.mp4', writer='ffmpeg', fps=1)
            plt.close('all')
            print(frames)
        return x