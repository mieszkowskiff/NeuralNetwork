import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("TkAgg")


def plot_classification(X, out, classes = 4):
    if classes == 4:
        colors = ['blue', 'darkred', 'red', 'mediumblue']
        legend_labels = ["correct 0 0", "incorrect 0 1", "incorrect 1 0", "correct 1 1"]
    if classes == 3:
        colors = ['blue', 'green', 'red']
        legend_labels = ["class 0", "class 1", "class 2"]
    if classes == 2:
        colors = ['blue', 'red']
        legend_labels = ["class 0", "class 1"]
    if classes == 9:
        colors = ['blue', 'darkred', 'red', 'firebrick', 'mediumblue', 'red', 'darkred', 'firebrick', 'royalblue']
        legend_labels = ["correct 0 0", 
                         "incorrect 0 1", 
                         "incorrect 0 2", 
                         "incorrect 1 0", 
                         "correct 1 1", 
                         "incorrect 1 2", 
                         "incorrect 2 0", 
                         "incorrect 2 1",
                         "correct 2 2"]
    cmap = mcolors.ListedColormap(colors)
    plt.scatter(X[0], X[1], c=out, cmap=cmap)
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                        markerfacecolor=color, markersize=10) 
            for label, color in zip(legend_labels, colors)]
    plt.legend(handles=handles, title='Values', loc='upper right')
    plt.title("Classification")
    plt.grid(True)
    plt.show()

