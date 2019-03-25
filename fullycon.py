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
import visdom

writer = tensorboardX.SummaryWriter()

# parameters
hidden_size = 500
batch_size = 1

# hyperparamters
epochs = 100
learning_rate = 0.0001  # decrease lr if loss increases, or increase lr if loss decreases.


def loader_data():
    # data loading: train and test
    trainX = hkl.load('dontPush/geno200X500.hkl')
    trainX = torch.from_numpy(trainX).float()
    trainY = pd.read_csv('dontPush/pheno200X500.csv', sep="\t")
    trainY = torch.tensor(trainY["f.4079.0.0"].values).float().unsqueeze(-1)

    # get means and vars
    x_mean = trainX.mean(dim=0)
    x_var = trainX.var()
    y_mean = trainY.mean()
    y_var = trainY.var()
    concatXY = np.column_stack((trainX, trainY))
    concatXY = torch.from_numpy(concatXY)

    # normalization for x and y
    trainX -= trainX.mean()
    trainX /= trainX.var()
    trainY -= trainY.mean()
    trainY /= trainY.var()
    concatXY -= concatXY.mean()
    concatXY /= concatXY.var()

    # build loader with x and y
    train = torch.utils.data.TensorDataset(trainX, trainY)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader, trainX.shape[1], trainY.shape[1], x_mean, x_var, y_mean, y_var


class FC(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.predict = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

    # def log_weights(self, step):
    #     for key, value in net.named_parameters():
    #         writer.add_histogram('Train/weights', value, epoch)


data_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = loader_data()
net = FC(x_features, y_features)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss()
testLoss = nn.MSELoss()
doesitwork = nn.L1Loss()


for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(data_loader):
        step = epoch * len(data_loader) + i
        inputs, labels = data

        prediction = net(inputs)

        if epoch == epochs -1:
            original_prediction = prediction * y_var + y_mean
            original_label = labels * y_var + y_mean
            print(original_prediction, original_label)
        # visual_loss = doesitwork(original_label, original_prediction)
        # writer.add_scalar('Train/doesitwork', visual_loss, step)


        l = criterion(prediction, labels)
        loss += l
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        #        writer.add_scalar('Train/weights', net.predict.weight[-1], epoch)

        # # Apply gradients
        # for param in net.parameters():
        #     param.data.add_(-learning_rate * param.grad.data)


        writer.add_scalar('Train/Loss', l.item(), step)
        # net.log_weights(step)


    writer.add_scalar('Epoch/Loss', loss, epoch)
    print('Epoch:  %d | Loss: %.4f ' % (epoch + 1, loss.item()))

# print(net.parameters())
# for key, value in net.named_parameters():
#     print(key)
