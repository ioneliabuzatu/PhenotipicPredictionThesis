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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import tensorboardX
import mlflow


writer = tensorboardX.SummaryWriter()

# parameters
output_size = 1 # predict one trait for now
input_size = 1 # tot snps or observations for sample
hidden_size = 50
batch_size = 200
obsevations_at_step1 = 1 # one at a time then change it to 300 --> corresponds to seq_len
num_layers = 1 # two layers rnn


# hyperparamters
epochs = 100
learning_rate = 0.001  # decrease lr if loss increases, or increase lr if loss decreases.


# data loading: train and test
trainX = hkl.load('dontPush/geno200X500.hkl')
trainX = torch.from_numpy(trainX).float()
trainY = pd.read_csv('dontPush/pheno200X500.csv', sep="\t")
trainY = torch.tensor(trainY["f.4079.0.0"].values).float()

#### for debuggin
concatXY = np.column_stack((trainX, trainY))
concatXY = torch.from_numpy(concatXY)
# print(concatXY.shape)
x = torch.ones_like(trainX)
y = torch.ones_like(trainY)
# print(x.shape, y.shape)
train = torch.utils.data.TensorDataset(trainY, trainY)
loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)  # train_loader ready
# for i, data in enumerate(loader):
#         inputs, labels = data
#         print(inputs.shape)
#


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


class Model(nn.Module):

    def __init__(self): # GRU, LSTM or RelU RNN
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, nonlinearity="relu") # order of the model paramters
        self.fc1 = nn.Linear(hidden_size, output_size)

    # this is what model.run calls
    def forward(self, x, hidden):
        # reshape input
        x = x.view(batch_size, 1, -1) # one snp at a time?
        # farward and back propagation
        out, hidden = self.rnn(x, hidden)
        # reshape output
        # out = out.view(-1, output_size)
        out = self.fc1(out[:, -1, :])
        return hidden, out

    # hidden states, zeros at first than thay change
    def init_hidden(self):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))

    def log_weights(self, step):
        for key, value in model.rnn.named_parameters():
            writer.add_histogram(key, value, step)
            # writer.add_scalar(key, value, step)


model = Model()

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=8)


# correct loop with right indentation
for epoch in range(epochs):
    # model.zero_grad()
    hidden = model.init_hidden() # initial hidden and cell states

    loss = 0

    for i, data in enumerate(loader):
        inputs, labels = data
        labels = Variable(labels)
        inputs = Variable(inputs.view(batch_size, 1, -1))

        # optimizer.zero_grad()

        hidden, output = model(inputs, hidden)

        l = criterion(output, labels)
        loss += l

        loss.backward()

        optimizer.step()

        step = epoch * len(loader) + i
        log_scalar('train_loss', loss.data.item(), step)
        model.log_weights(step)
    # lr_scheduler.step(loss)
    print('Epoch:  %d | Loss: %.4f' % (epoch + 1, loss.item()))


# torch.save(model.state_dict(), 'model.ckpt')


# print(model.rnn.all_weights)
# w = list(model.parameters())
# print(w)