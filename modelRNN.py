# import os
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
import tensorboardX
from dataLoaders import train_data


writer = tensorboardX.SummaryWriter()

# hyperparameters
hidden_size = 50
batch_size = 5
epochs = 100
learning_rate = 0.1


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, tot_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # num of hidden dim
        self.tot_layers = tot_layers  # num of hidden layers
        self.rnn = nn.RNN(hidden_size, output_dim, batch_first=True, nonlinearity="tanh")
        self.fc1 = nn.Linear(input_size, hidden_size)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))

        # processing feed-forward layers first improves learning
        first_fc = self.fc1(input)
        out, h0 = self.rnn(first_fc, h0)
        # rnn_out, h0 = self.rnn(input, h0)
        # out = self.fc1(rnn_out[:, -1, :])

        return out


train_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
net = RNN(x_features, y_features)
criterion = nn.SmoothL1Loss()
doesitwork = nn.MSELoss()
optimizer = optim.Adam(net.parameters())


for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(train_loader):
        step = epoch * len(train_loader) + 1
        inputs, labels = data

        inputs = Variable(inputs.view(batch_size, 1, -1))
        labels = Variable(labels)

        prediction = net(inputs)

        original_prediction = prediction * y_var + y_mean
        original_label = labels * y_var + y_mean
        # if epoch == epochs -1:
        #     print(original_prediction, original_label)

        visual_loss = doesitwork(original_label, original_prediction)
        writer.add_scalar('Train/doesitwork', visual_loss, step)

        l = criterion(prediction, labels)
        loss += l

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        writer.add_scalar('Train/l', l.item(), step)

    writer.add_scalar('Epoch/Loss', loss, epoch)
    print('Epoch:  %d | Loss: %.4f' % (epoch + 1, loss))


"""
net.load_state_dict(torch.load('model_RNN.ckpt'))

with torch.no_grad():
    test_loss_tot = 0
    accuracy = 0
    i = 0
    for id, pressure in test_loader:

        id = id.view(batch_size, 1, -1)

        test_prediction = net(id)

        original_prediction = test_prediction * y_var + y_mean
        original_label = pressure * y_var + y_mean

        test_loss = criterion(original_prediction, pressure)
        test_loss_tot += test_loss

        # plot test loss
        writer.add_scalar('Test/Loss', test_loss, i)

        i+=1 # needed for plotting loss at each step

        print(original_prediction, pressure)

    print("test loss is {}".format(test_loss_tot))


# Save and load only the model parameters
# torch.save(net.state_dict(), 'paramsRNN.ckpt')
# net.load_state_dict(torch.load('paramsRNN.ckpt'))
"""