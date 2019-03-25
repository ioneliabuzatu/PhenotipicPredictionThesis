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

writer = tensorboardX.SummaryWriter()

# hyperparameters
hidden_size = 50
batch_size = 1
epochs = 100
learning_rate = 0.05


def loader_data():
    # data loading: train and test
    trainX = hkl.load('dontPush/geno200X500.hkl')
    trainX = torch.from_numpy(trainX).float()
    trainY = pd.read_csv('dontPush/pheno200X500.csv', sep="\t")
    trainY = torch.tensor(trainY["f.4079.0.0"].values).float().unsqueeze(-1)
    y_mean = trainY.mean()
    y_var = trainY.var()
    concatXY = np.column_stack((trainX, trainY))
    concatXY = torch.from_numpy(concatXY)
    x_mean = trainX.mean(dim=0)
    x_var = trainX.var(dim=0)

    # normalization for x and y
    trainX -= trainX.mean(dim=0)
    trainX /= trainX.var(dim=0) + 1e-10
    trainY -= trainY.mean()
    trainY /= trainY.var()
    concatXY -= concatXY.mean(dim=0)
    concatXY /= concatXY.var(dim=0)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader, trainX.shape[1], trainY.shape[1], x_mean, x_var, y_mean, y_var


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, tot_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # num of hidden dim
        self.tot_layers = tot_layers  # num of hidden layers
        self.rnn = nn.LSTM(hidden_size, output_dim, batch_first=True, nonlinearity="tanh")
        self.fc1 = nn.Linear(input_size, hidden_size)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))

        # processing feed-forward layers first improves learning
        first_fc = self.fc1(input)
        out, h0 = self.rnn(first_fc, h0)
        # rnn_out, h0 = self.rnn(input, h0)
        # out = self.fc1(rnn_out[:, -1, :])

        return out


data_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = loader_data()
net = RNN(x_features, y_features)
criterion = nn.SmoothL1Loss()
doesitwork = nn.MSELoss()
optimizer = optim.Adam(net.parameters())


for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(data_loader):
        step = epoch * len(data_loader) + 1
        inputs, labels = data

        inputs = Variable(inputs.view(batch_size, 1, -1))
        labels = Variable(labels)

        prediction = net(inputs)

        original_prediction = prediction * y_var + y_mean
        original_label = labels * y_var + y_mean
        if epoch == epochs -1:
            print(original_prediction, original_label)

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


