import numpy as np


class Loss:
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        pass

    def gradient(self, y_true, y_pred):
        pass


class Softmax(Loss):
    def __init__(self):
        # Will be used by gradient.
        self.probs = None

    def loss(self, y_true, y_pred):
        # This is just for preventing probs from exploding in exp terms.
        self.probs = None
        y_pred -= np.max(y_pred, axis=1, keepdims=True)
        probs = np.exp(y_pred)
        probs /= np.sum(probs, axis=1, keepdims=True)
        self.probs = probs
        N = y_true.shape[0]
        # print(N)
        correct_probs = probs[np.arange(N), y_true]
        loss = np.sum(-np.log(correct_probs)) / N
        return loss

    def gradient(self, y_true, y_pred):
        N = y_true.shape[0]
        grad = self.probs
        grad[np.arange(N), y_true] -= 1
        grad /= N
        return grad


class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoiding division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoiding division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
