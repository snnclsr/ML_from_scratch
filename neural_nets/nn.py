import numpy as np
from layers import Layer


def batch_generator(X, y, batch_size):
    M = X.shape[0]
    for i in range(0, M, batch_size):
        yield X[i: i + batch_size], y[i: i + batch_size]


class NeuralNet:
    def __init__(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

        self.layers = []

    def add(self, layer):
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        self.layers.append(layer)

    def forward_prop(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backprop(self, grad):
        out_grad = grad
        for layer in reversed(self.layers):
            out_grad = layer.backward(out_grad)

    def train(self, X, y, n_iter, batch_size=32, print_verbose=10):
        train_loss = []
        for i in range(n_iter):
            losses = []
            for X_batch, y_batch in batch_generator(X, y, batch_size):
                preds = self.forward_prop(X_batch)
                l = self.loss.loss(y_batch, preds)
                losses.append(l)
                grad = self.loss.gradient(y_batch, preds)
                self.backprop(grad)
                # self.optimizer.step(param, dparam)
            train_loss.append(np.mean(losses))
            if i % print_verbose == 0:
                print("Iteration : {}, Loss : {}".format(i, train_loss[-1]))

        return train_loss

    def predict(self, X):
        preds = self.forward_prop(X)
        return np.argmax(preds, axis=1)
