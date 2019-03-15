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
input_size = 300
hidden_size = 300
batch_size = 300
epochs = 5
learning_rate = 0.5


# loading data into train and test
trainX = hkl.load('dontPush/bigTraining.hkl')
trainX = torch.from_numpy(trainX).float()
trainY = pd.read_csv('dontPush/pheno10000.csv', sep="\t")
trainY = torch.tensor(trainY["f.4079.0.0"].values).float()
train = torch.utils.data.TensorDataset(trainX, trainY)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)  # train_loader ready


features_test = hkl.load("dontPush/test100.hkl")
features_test = torch.from_numpy(features_test).float()
targets_test = pd.read_csv("dontPush/pheno100.csv", sep="\t")
targets_test = torch.tensor(targets_test['f.4079.0.0'].values).type(torch.LongTensor)
test = torch.utils.data.TensorDataset(features_test, targets_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=300, shuffle=False, drop_last=True) # test_loader ready


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, tot_layers=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size # num of hidden dim
        self.tot_layers = tot_layers # num of hidden layers
        self.rnn = nn.RNN(input_size, hidden_size, tot_layers, batch_first=True, nonlinearity="tanh")  # change it to tanh

        self.fc1 = nn.Linear(hidden_size, output_dim)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))
        rnn_out, hn = self.rnn(input, h0) # lstm = nn.LSTM(10000, 300)
        out = self.fc1(rnn_out[:, -1, :])
        return out


def training(model, train_loader, optimiser, loss_func):
    loss_tot = []
    for epoch in range(epochs):
        for i, (person, bloodP) in enumerate(train_loader):
            trainX = Variable(person.view(batch_size, 1, -1))
            targets = Variable(bloodP)
            # print(targets.size())
            optimiser.zero_grad()
            outputs = model(trainX)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimiser.step()
            loss_tot.append(loss.item())
            writer.add_scalar('data/loss', loss, i + (epoch * len(train_loader)))
        # print("epoch {}, loss {}".format(epoch, loss_tot.pop()))


def testing(model, train_loader):
    for i, (persons, bloodP) in enumerate(train_loader):
        input = Variable(persons.view(batch_size, 1, -1))
        outputs = model(input)
        print(outputs)


def main(train_loader = train_loader, test_loader = test_loader):
    model = RNN(input_size, hidden_size)

    loss1 = torch.nn.MSELoss()
    loss2 = nn.SmoothL1Loss()
    optimiser1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimiser2 = optim.SGD(model.parameters(), lr=learning_rate)

    trained = training(model, train_loader, optimiser = optimiser1, loss_func = loss1)
    # tested = testing(model, test_loader)

    for i, (persons, bloodP) in enumerate(test_loader):
        persons = Variable(persons.view(batch_size, 1, -1))
        outputs = model(persons)
        print(outputs)


if __name__ == "__main__":
    main()



"""Notes: about RNNs
the probability of keeping the context from a word that is far away from the current word being processed decreases exponentially with the distance from it.
"""