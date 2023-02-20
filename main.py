import csv
import random

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i])for i in range(len(iris.target))]



f = open("Iris.csv").readlines()

input_dim = 4
out_dim = 3
h_dim = 10

x = np.random.randn(1, input_dim)
y = np.random.randn(0, out_dim-1)

w1 = np.random.randn(input_dim, h_dim)
b1 = np.random.randn(h_dim)
w2 = np.random.randn(h_dim, out_dim)
b2 = np.random.randn(out_dim)

def toFull(y, num_classes):
    y_full = np.zeros((1,num_classes))
    y_full[0, y] = 1
    return y_full

def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out/np.sum(out)

def relu_deriv(t):
    return (t>=0).astype(float)

ALPHA = 0.0002
EPOCHS = 500
loss_arr = []
for ep in range(EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset)):

        x, y = dataset[i]

        #Forward
        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax(t2)
        E = sparse_cross_entropy(z, y)

        #Baclward
        y_full = toFull(y, out_dim)
        de_dt2 = z -y_full
        de_dw2 = h1.T @ de_dt2
        de_db2 = de_dt2
        de_dh1 = de_dt2 @ w2.T
        de_dt1 = de_dh1 * relu_deriv(t1)
        de_dw1 = x.T @ de_dt1
        de_db1 = de_dt1

        #Update
        w1 = w1 - ALPHA*de_dw1
        b1 = b1 - ALPHA*de_db1
        w2 = w2 - ALPHA*de_dw2
        b2 = b2 - ALPHA*de_db2

        loss_arr.append(E)

def predict(x):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z

def calc_acc():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct+=1
    acc = correct/len(dataset)
    return acc

acc = calc_acc()
print("Acc: ", acc)

# import matplotlib.pyplot as plt
# plt.plot(loss_arr)
# plt.show()

q = np.array([6.2, 2.2, 4.5, 1.5])
probs = predict(q)
pred_class = np.argmax(probs)
class_names = ['Setosa', 'Versicolor', 'Virginica']
print(class_names[pred_class])
