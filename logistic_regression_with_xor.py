"""
@author = mbilkhu
Code for Logistic Regression
"""

import numpy as np

def load_dataset():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    W = np.random.randn(1,2)
    b = np.ones(1)
    x = X.T
    y = y.T
    print("X Shape:- " +  str(x.shape))
    print("Y Shape:- " +  str(y.shape))
    return x,y,W,b

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(x):
    return x * (1-x)

def forward_propagate(W, x, b):
    z = np.dot(W, x) + b
    a = sigmoid(z)
    return a

def compute_loss(y, a):
    x = np.multiply(y, np.log(a)) + np.multiply((1-y), np.log(1-a))
    cost = np.sum(x, dtype=np.float32)
    return -np.squeeze(cost)

def gradient_descent(y, a, x):
    dz = (a-y) * sigmoid_gradient(a)
    dw = np.dot(dz, x.T)
    db = dz
    return dw, db

def main():
    x,y,W,b = load_dataset()
    loss = []
    n_epochs = 60000
    learning_rate = 0.01
    for i in range(n_epochs):
        a = forward_propagate(W, x, b)
        cost = compute_loss(y, a)
        loss.append(cost)
        dw, db = gradient_descent(y, a, x)
        W = W - learning_rate * dw
        b = b - learning_rate * db
        if i%250 == 0:
            print("Loss after epoch %d = %f " % (i, cost))
    print(a)

if __name__=='__main__':
    main()
