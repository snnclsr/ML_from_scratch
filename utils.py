import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_data(X, y, s=25, cmap=cmap_bold):
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap, s=s)


def plot_decision_boundary(model, X, y, cmap=cmap_bold, alpha=0.6):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=alpha)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
