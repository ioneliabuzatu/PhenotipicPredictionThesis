import torch
import hickle as hkl
import pandas as pd
import numpy as np
import torch.utils.data
from config import train_geno, train_pheno, test_geno, test_pheno

def train_data():
    # data loading: train and test
    trainX = hkl.load(train_geno)
    trainX = torch.from_numpy(trainX).float()
    trainY = pd.read_csv(train_pheno, sep="\t")
    trainY = torch.tensor(trainY["f.4079.0.0"].values).float().unsqueeze(-1)
    y_mean = trainY.mean()
    y_var = trainY.var()
    concatXY = np.column_stack((trainX, trainY))
    concatXY = torch.from_numpy(concatXY)
    x_mean = trainX.mean(dim=0)
    x_var = trainX.var(dim=0)

    # normalization for x and y
    trainX -= trainX.mean(dim=0)
    trainX /= trainX.var(dim=0) + 1e-10 # a small number is added to prevent nan when sd == 0
    trainY -= trainY.mean()
    trainY /= trainY.var()
    concatXY -= concatXY.mean(dim=0)
    concatXY /= concatXY.var(dim=0)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    # TODO: think of a way to import only once batch_size evrywhere
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, trainX.shape[1], trainY.shape[1], x_mean, x_var, y_mean, y_var


def test_data():
    testX = hkl.load(test_geno)
    testX = torch.from_numpy(testX).float()
    testY = pd.read_csv(test_pheno, sep="\t")
    testY = torch.tensor(testY["f.4079.0.0"].values).float().unsqueeze(-1)

    # get means and vars
    x_mean = testX.mean(dim=0)
    x_var = testX.var()
    y_mean = testY.mean()
    y_var = testY.var()

    # normalization for x and y
    testX -= testX.mean()
    testX /= testX.var()
    # testY -= testY.mean()
    # testY /= testY.var()

    test = torch.utils.data.TensorDataset(testX, testY)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return test_loader, x_mean, x_var, y_mean, y_var

train_data()
test_data()