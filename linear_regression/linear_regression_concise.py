import sys
sys.path.append("../utils")
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
from d2l import torch as d2l

from utils import generate_synthetic_data


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, 1.2, -5.2, 100.32])
true_b = -90.18
features, labels = generate_synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array([features, labels], batch_size)
print(next(iter(data_iter)))
net = nn.Sequential(nn.Linear(4, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss(reduction='mean')
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"Epoch {epoch}: loss = {l}")
