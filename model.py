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
# import tensorboardX


# parameters
output_size = 1 # predict one trait for now
input_size = 301 # tot snps or observations for sample
hidden_size = 300
batch_size = 300
obsevations_at_step1 = 1 # one at a time then change it to 300 --> corresponds to seq_len
num_layers = 1 # two layers rnn


# hyperparamters
epochs = 100
learning_rate = 0.1


# data loading: train and test
trainX = hkl.load('dontPush/bigTraining.hkl')
trainX = torch.from_numpy(trainX).float()
trainY = pd.read_csv('dontPush/pheno10000.csv', sep="\t")
trainY = torch.tensor(trainY["f.4079.0.0"].values).float()

#### for debuggin
concatXY = np.column_stack((trainX, trainY))
concatXY = torch.from_numpy(concatXY)
print(concatXY.shape)
x = torch.ones_like(trainX)
y = torch.ones_like(trainY)
# print(x.shape, y.shape)
train = torch.utils.data.TensorDataset(concatXY, trainY)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)  # train_loader ready


class Model(nn.Module):

    def __init__(self): # GRU, LSTM or RelU RNN
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True) # order of the model paramters

    # this is what model.run calls
    def forward(self, x, hidden):
        # reshape input
        x = x.view(batch_size, 1, -1) # one snp at a time?
        # farward and back propagation
        out, hidden = self.rnn(x, hidden)
        # reshape output
        out = out.view(-1, output_size)

        return hidden, out

    # hidden states, zeros at first than thay change
    def init_hidden(self):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


model = Model()

loss1 = torch.nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        labels = Variable(labels)
        inputs = Variable(inputs.view(batch_size, 1, -1))
        hidden, output = model(inputs, hidden)

        loss += criterion(output, labels)
    loss.backward()
    optimizer.step()

    print('Epoch:  %d | Loss: %1.3f' % (epoch + 1, loss.item()))


