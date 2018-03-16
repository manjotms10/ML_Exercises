"""
@author = mbilkhu
Code for a Shallow Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from random import shuffle

def load_dataset():
    np.random.seed(0)
    X = datasets.load_iris().data
    y = datasets.load_iris().target
    W1 = np.random.randn(50,4)
    b1 = np.ones((50,1))
    W2 = np.random.randn(30,50)
    b2 = np.ones((30,1))
    W3 = np.random.randn(10, 30)
    b3 = np.ones((10, 1))
    W4 = np.random.randn(3,10)
    b4 = np.ones((3,1))
    X = X.T
    y = np.eye(3)[y]
    y = y.T
    print("X Shape:- " +  str(X.shape))
    print("Y Shape:- " +  str(y.shape))
    params = {"x": X, "y": y, "W1": W1, "W2": W2, "W3": W3, "W4": W4, "b1": b1, "b2": b2, "b3": b3, "b4": b4}
    return params

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def softmax(x):
    x -= np.max(x)
    x = np.exp(x)/np.sum(np.exp(x), axis=0)[None, :]
    return x

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
    params = load_dataset()

    # Extract Parameters
    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]
    W4 = params["W4"]
    b1 = params["b1"]
    b2 = params["b2"]
    b3 = params["b3"]
    b4 = params["b4"]
    x = params["x"]
    y = params["y"]
    loss = []

    n_epochs = 30000
    learning_rate = 0.0001

    for i in range(n_epochs):
        # --- Forward Pass ---
        z1 = np.dot(W1, x) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = np.tanh(z2)
        z3 = np.dot(W3, a2) + b3
        a3 = np.tanh(z3)
        z4 = np.dot(W4, a3) + b4
        a4 = sigmoid(z4)
        a4 = softmax(a4)

        # --- Compute Loss ---
        cost = compute_loss(y, a4)
        loss.append(cost)

        # --- Backprop and parameter update ---
        dz4, dw4, db4 = gradient_descent(y, a4, a3)
        W4 = W4 - learning_rate * dw4
        b4 = b4 - learning_rate * db4
        dz3, dw3, db3 = gradient_descent_middle_layers(W4, dz4, z3, a3, a2)
        W3 = W3 - learning_rate * dw3
        b3 = b3 - learning_rate * db3
        dz2, dw2, db2 = gradient_descent_middle_layers(W3, dz3, z2, a2, a1)
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
    print(a4[:, 72:74])
    plt.show()

if __name__=='__main__':
    main()
