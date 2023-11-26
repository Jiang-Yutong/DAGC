'''
Simple CNN for MNIST dataset, the figure in MNIST has 1 channel

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __deepcopy__(self):
        new_model = CNN1(copy.deepcopy(self.conv1), self.classifier[0].in_features, self.classifier[-1].out_features)

        for i, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                new_model.classifier[i].weight = copy.deepcopy(layer.weight)
                new_model.classifier[i].bias = copy.deepcopy(layer.bias)
        return new_model

    def copy(self):
        return copy.deepcopy(self)


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x