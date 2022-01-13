import sys, os
sys.path.append('../utils')
import random
import torch
from d2l import torch as d2l
from utils import generate_synthetic_data, data_iter

# def generate_synthetic_data(w, b, num_examples):
    # x = torch.normal(0, 1, (num_examples, len(w)))
    # y = torch.matmul(x, w) + b
    # y += torch.normal(0, 0.01, y.shape)
    # return x, y.reshape(-1, 1)


true_w = torch.tensor([2, -3.4, 1.1])
true_b = 1.3
features, labels = generate_synthetic_data(true_w, true_b, 10000)
print(f"Features = {features[0]}\nLabels = {labels[0]}")


# def data_iter(batch_size, features, labels):
    # num_examples = len(features)
    # indices = list(range(num_examples))
    # random.shuffle(indices)
    # for i in range(0, num_examples, batch_size):
        # batch_indices = torch.tensor(
            # indices[i: min(i + batch_size, num_examples)]
        # )
        # yield features[batch_indices], labels[batch_indices]


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(f"{X}\n{y}")
    break
w = torch.normal(0, 0.01, size=(3, 1), requires_grad=True)
# w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linear_regression(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_pred, y):
    return (y_pred - y)**2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr, num_epochs = 0.1, 10
net, loss = linear_regression, squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"Epoch {epoch}: Loss = {train_l.mean()}")

print(f"Error estimating w = {true_w - w.reshape(true_w.shape)}")
print(f"Error in bias = {true_b - b}")

