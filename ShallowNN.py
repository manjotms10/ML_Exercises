"""
@author = mbilkhu
Code for a Shallow Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from random import shuffle

def load_dataset():
    X = datasets.load_iris().data
    y = datasets.load_iris().target
    W1 = np.random.randn(20,4)
    b1 = np.ones((20,1))
    W2 = np.random.randn(3,20)
    b2 = np.ones((3,1))
    X = X.T
    y = np.eye(3)[y]
    y = y.T
    print("X Shape:- " +  str(X.shape))
    print("Y Shape:- " +  str(y.shape))
    return X,y,W1,b1,W2,b2

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(x):
    return x * (1-x)

def forward_propagate(W, x, b):
    z = np.dot(W, x) + b
    return z

def compute_loss(y, a):
    x = np.multiply(y, np.log(a)) + np.multiply((1-y), np.log(1-a))
    cost = np.sum(x, dtype=np.float32)
    return -np.squeeze(cost)

def gradient_descent(y, a, x):
    dz = (a-y) * sigmoid_gradient(a)
    dw = np.dot(dz, x.T)
    db = dz
    return dz, dw, db

def gradient_descent_middle_layers(W1,dz1,z,a1, x):
    x1 = np.dot(W1.T, dz1)
    x2 = 1 - np.power(a1,2)
    dz = np.multiply(x1, x2)
    dw = np.dot(dz, x.T)/150
    db = np.sum(dz, axis=1, keepdims=True)
    return dz, dw, db

def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = shuffle(X, y)

    for i in range(0, X.shape[1], minibatch_size):
        X_mini = X[0][i:i + minibatch_size]
        y_mini = y[0][i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches

def main():
    x,y,W1,b1,W2,b2 = load_dataset()
    loss = []
    n_epochs = 60000
    learning_rate = 0.00001

    for i in range(n_epochs):
        z1 = np.dot(W1, x) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)

        cost = compute_loss(y, a2)
        loss.append(cost)

        dz2, dw2, db2 = gradient_descent(y, a2, a1)
        W2 = W2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2
        dz1, dw1, db1 = gradient_descent_middle_layers(W2, dz2, z1, a1, x)
        W1 = W1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        if i%500 == 0:
            print("Loss after epoch %d = %f " % (i, cost))
    plt.xlabel("#Iterations")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()

if __name__=='__main__':
    main()
