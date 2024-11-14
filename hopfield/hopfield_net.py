import numpy as np
import matplotlib.pyplot as plt
import display
from matplotlib.animation import FuncAnimation
import copy

def heavy_side(x):
    return np.heaviside(x, 0)

def signum(x):
    return 2*heavy_side(x)-1

class HopfieldNet:
    def __init__(self, n, activation, dynamics):
        self.n = n
        self.dynamics=dynamics

        self.W = np.zeros((n, n), dtype=np.float32)

        if activation == 'signum':
            self.activation = signum
        elif activation == 'heaviside':
            self.activation = heavy_side

    def HEBB_training(self, X):
        for x in X:
            self.W += np.outer(x, x)
        np.fill_diagonal(self.W, 0)
        self.W /= self.n

    def OJA_training(self, X):
        for x in X:
            self.W += np.outer(x, x)
        np.fill_diagonal(self.W, 0)
        self.W /= self.n
        self.W -= np.diag(np.diag(self.W))


    def call(self, x):
        if self.dynamics=='asynchronous':
            for i in range(self.n):
                u = np.dot(self.W[i], x)
                x[i] = self.activation(u)
        elif self.dynamics=='synchronous':
            u = np.dot(self.W, x)
            x = self.activation(u)
        return x

    def forward(self, dims, init_x, epochs, animation = False):
        x = np.array(init_x)
        if animation:
            frames = [copy.deepcopy(x)]
        display.display(x, dims)
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