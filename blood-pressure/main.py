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
# hyperparameters
hidden_size = 500
batch_size = 1
epochs = 100
learning_rate = 0.5


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
    x_var = trainX.var()

    # normalization for x and y
    trainX -= trainX.mean(dim=0)
    trainX /= trainX.var()
    trainY -= trainY.mean()
    trainY /= trainY.var()
    concatXY -= concatXY.mean()
    concatXY /= concatXY.var()

    train = torch.utils.data.TensorDataset(trainX, trainY)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader, trainX.shape[1], trainY.shape[1], x_mean, x_var, y_mean, y_var

data_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = loader_data()
net = RNN(x_features, y_features)
criterion = nn.SmoothL1Loss()
doesitwork = nn.MSELoss()
optimizer = optim.Adam(net.parameters())