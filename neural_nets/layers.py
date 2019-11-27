import numpy as np

import copy


class Layer:
    def __init__():
        pass

    def forward(self, X):
        pass

    def backward(self, grad_top):
        pass


class Dense(Layer):
    def __init__(self, input_size, out_size):
        self.input_size = input_size
        self.out_size = out_size
        self.layer_input = None

        # self.W = np.random.randn(input_size, out_size) * 0.01
        # Glorot uniform initalization
        limit = np.sqrt(6) / np.sqrt(input_size + out_size)
        self.W = np.random.uniform(-limit, limit, size=(input_size, out_size))
        self.b = np.zeros((1, out_size))

    def initialize(self, optimizer):
        self.W_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def forward(self, X):
        self.layer_input = X
        return np.dot(X, self.W) + self.b

    def backward(self, out_grad):

        dW = self.layer_input.T.dot(out_grad)
        db = np.sum(out_grad, axis=0, keepdims=True)
        out_grad = out_grad.dot(self.W.T)

        self.W = self.W_optimizer.step(self.W, dW)
        self.b = self.b_optimizer.step(self.b, db)

        return out_grad


class Dropout(Layer):
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None

    def forward(self, X, is_training=True):
        if is_training:
            self.mask = (np.random.uniform(size=X.shape) > self.p) / self.p
            return X * self.mask
        return X

    def backward(self, grad_top):
        return self.mask * grad_top


class BatchNormalization(Layer):
    def __init__(self, momentum=0.9, eps=1e-5):
        self.momentum = momentum
        self.eps = eps
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
        self.X_norm = None
        self.denum = None

    def initialize(self, optimizer):
        self.g_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def forward(self, X, is_training=True):

        if is_training:
            mb_mean = np.mean(X, axis=0)
            mb_var = np.var(X, axis=0)
            # First time running through batch norm layer.
            if self.running_mean is None:
                self.running_mean = np.zeros(X.shape[1], dtype=X.dtype)
                self.running_var = np.zeros(X.shape[1], dtype=X.dtype)

                self.gamma = np.ones(X.shape[1])
                self.beta = np.zeros(X.shape[1])

            denum = (1 / np.sqrt(mb_var + self.eps))
            X_norm = (X - mb_mean) / np.sqrt(mb_var + self.eps)

            # Parameters that will be used by backpropagation.
            self.X_norm = X_norm
            self.denum = denum
            # Exponential decay.
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mb_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * mb_var
            return self.gamma * X_norm + self.beta

        # Test time
        else:
            centered_X = X - self.running_mean
            normalized_X = centered_X / np.sqrt(self.running_var + self.eps)
            return self.gamma * normalized_X + self.beta

    def backward(self, out_grad):
        dbeta = np.sum(out_grad, axis=0)
        dgamma = np.sum(out_grad * self.X_norm, axis=0)

        N = out_grad.shape[0]
        dx_norm = out_grad * self.gamma
        dx = (1 / N) * self.denum * (N * dx_norm - np.sum(dx_norm, axis=0) -
                                     self.X_norm * np.sum(dx_norm * self.X_norm, axis=0))

        self.gamma = self.g_optimizer.step(self.gamma, dgamma)
        self.beta = self.b_optimizer.step(self.beta, dbeta)

        return dx
