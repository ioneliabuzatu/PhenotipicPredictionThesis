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

# writer = tensorboardX.SummaryWriter()

# hyperparameters
input_size = 301
hidden_size = 300
batch_size = 300
epochs = 100
learning_rate = 0.005


# loading data into train and test
trainX = hkl.load('dontPush/bigTraining.hkl')
trainX = torch.from_numpy(trainX).float()
trainY = pd.read_csv('dontPush/pheno10000.csv', sep="\t")
trainY = torch.tensor(trainY["f.4079.0.0"].values).float()

#### for debuggin
concatXY = np.column_stack((trainX, trainY))
concatXY = torch.from_numpy(concatXY)
# print(concatXY.shape)
x = torch.ones_like(trainX)
y = torch.ones_like(trainY)
train = torch.utils.data.TensorDataset(concatXY,trainY)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)  # train_loader ready


features_test = hkl.load("dontPush/test100.hkl")
features_test = torch.from_numpy(features_test).float()
targets_test = pd.read_csv("dontPush/pheno100.csv", sep="\t")
targets_test = torch.tensor(targets_test['f.4079.0.0'].values).type(torch.LongTensor)
test = torch.utils.data.TensorDataset(features_test, targets_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True) # test_loader ready


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, tot_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size # num of hidden dim
        self.tot_layers = tot_layers # num of hidden layers
        self.rnn = nn.RNN(input_size, hidden_size,  batch_first=True, nonlinearity="tanh")
        self.fc1 = nn.Linear(hidden_size, output_dim)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))
        rnn_out, h0 = self.rnn(input, h0) # lstm = nn.LSTM(10000, 300)
        out = self.fc1(rnn_out[:, -1, :])  # why am I using a fully connected here?

        return out

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = torch.eq(logit, target).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


def train(model, train_loader, optimizer, criterion):
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()

        # epoch round
        for i, data in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs
            inputs, labels = data
            labels = Variable(labels)
            inputs = Variable(inputs.view(batch_size, 1, -1))

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)


            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, batch_size)

        model.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
              % (epoch+1, train_running_loss / i, train_acc / i))


# TODO: something wrong with the test
def testing(model, test_loader):
    for i, (persons, bloodP) in enumerate(test_loader):
        input = Variable(persons.view(batch_size, 1, -1))
        outputs = model(input)
        print(outputs)


def main(train_loader = train_loader, test_loader = test_loader):
    model = RNN(input_size, hidden_size)
    loss1 = torch.nn.MSELoss()
    loss2 = nn.SmoothL1Loss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer2 = optim.SGD(model.parameters(), lr=learning_rate)

    training = train(model, train_loader, optimizer = optimizer1, criterion = loss2)
    # tested = testing(model, test_loader)

    # for param in model.parameters():
    #     print(param.data)

if __name__ == "__main__":
    main()


