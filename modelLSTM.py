import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import torch.utils.data

x = hkl.load('dontPush/bigTraining.hkl')
# x = torch.from_numpy(x).float()
x = torch.from_numpy(x).float()

y = pd.read_csv('dontPush/pheno10000.csv', sep="\t")
y = torch.tensor(y["f.4079.0.0"].values).float()
print(x[32].shape)
print(x[33].shape)
print(x[34].shape)
print(x[35].shape)
# print(y.shape)


input_size = 300
hidden_size = 300
batch_size = 300
epochs = 50
learning_rate = 0.05


train = torch.utils.data.TensorDataset(x,y)
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
# print(type(train_loader))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, tot_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size # num of hidden dim
        self.tot_layers = tot_layers # num of hidden layers
        self.rnn = nn.RNN(input_size, hidden_size, tot_layers, batch_first=True, nonlinearity="relu")  # change it to tanh

        self.fc1 = nn.Linear(hidden_size, output_dim)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))
        rnn_out, hn = self.rnn(input, h0) # lstm = nn.LSTM(10000, 300)
        out = self.fc1(rnn_out[:, -1, :])
        return out


model = RNN(input_size, hidden_size)

loss_func = torch.nn.MSELoss()
optimiser1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer2 = optim.SGD(model.parameters(), lr=learning_rate)

# check weights before training
# with torch.no_grad():
#     test_input = x[0][:] # one single data point before training
#     output = model(test_input)
#     print(output)


loss_tot = []
for epoch in range(epochs):
    for i, (person, bloodP) in enumerate(train_loader):
        trainX = Variable(person.view(300, 1, -1))
        xShape = trainX.size(0)
        targets = Variable(bloodP[i])
        optimiser1.zero_grad()
        outputs = model(trainX)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimiser1.step()
        # loss_tot.append(loss.item())
        print(loss.item()) # TODO: RuntimeError: input.size(-1) must be equal to input_size. Expected 300, got 100
    # print("epoch {}, loss {}".format(epoch, loss_tot.pop()))


def main():
    pass