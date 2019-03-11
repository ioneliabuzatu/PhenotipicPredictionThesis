import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl


x = hkl.load('dontPush/bigTraining.hkl')
x = torch.from_numpy(x).float()
print(x.shape)
print(x[0][:])

y = pd.read_csv('dontPush/pheno10000.csv', sep="\t")
y = torch.tensor(y["f.4079.0.0"].values).float()

input_size = 300
hidden_size = 2
batch_size = 1000

learning_rate = 0.05

class LSTMbloodPressure(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, output_dim=1, tot_layers = 2): # let's try with 2
        super(LSTMbloodPressure, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tot_layers = tot_layers
        self.batch_size = batch_size

        # lst layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.tot_layers)
        # output layer
        self.linear  =  nn.Linear(self.hidden_size, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.tot_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.tot_layers, self.batch_size, self.hidden_size))

    def forward(self, input):
        # shae of output lstm layer (input_size, batch_size, hidden_size)
        # shape of self.hidden (a, b) where a and b both have
        # shape tot_size, batch_size, hidden_size)
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size), self.hidden)
        prediction = self.linear(lstm_out[-1].view(self.batch_size), -1)
        return prediction.view[-1]


model = LSTMbloodPressure(input_size, hidden_size, batch_size)

loss_func = loss_fn = torch.nn.MSELoss()
optimiser1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer2 = optim.SGD(model.parameters(), lr=learning_rate)

# check weights before training
with torch.no_grad():
    test_input = x[0][:] # one single data point before trianing
    output = model(test_input)
    print(output)


def main():
    pass