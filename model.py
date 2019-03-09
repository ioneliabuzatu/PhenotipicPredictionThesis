import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# hyperparameters
input_size = 100
hidden_size = 100
hidden2_size = 100
output_size = 2 # TODO: how many?

epochs = 20
batch_size = 50
learning_rate = 0.05
momentum = 0

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.activation1 = nn.ReLU()

        # TODO: return classifier fro traits
    def forward(self, input):
        input = self.fc1(input)
        input = self.activation1(input)
        input = self.fc2(input)
        input = self.activation1(input)
        input = self.fc3(input)
        return input

model = Net()


def train(criterion = nn.CrossEntropyLoss(), optimizer = optim.SGD(model.parameters()), lr = learning_rate, momentum = momentum):
    pass


def test():
    pass


def main():
    pass