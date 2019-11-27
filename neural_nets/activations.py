import numpy as np
from layers import Layer


class ReLU(Layer):
    def __init__(self):
        self.layer_input = None

    def forward(self, X):
        self.layer_input = X
        return np.maximum(0, X)

    def backward(self, out_grad):
        grad = np.where(self.layer_input > 0, 1, 0).astype(np.float64)
        grad = grad * out_grad
        return grad
