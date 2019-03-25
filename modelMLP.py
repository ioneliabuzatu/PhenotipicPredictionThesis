import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

# hyperparameters
input_size = 1
hidden_size = 50
hidden2_size = 200
output_size = 1 # TODO: how many?

epochs = 50
batch_size = 5
learning_rate = 0.05
momentum = 0
criterion = nn.MSELoss()


# data loading: train and test
trainX = hkl.load('dontPush/geno200X500.hkl')
trainX = torch.from_numpy(trainX).float()
trainY = pd.read_csv('dontPush/pheno200X500.csv', sep="\t")
trainY = torch.tensor(trainY["f.4079.0.0"].values).float()

#### for debugging
concatXY = np.column_stack((trainX, trainY))
concatXY = torch.from_numpy(concatXY)
# print(concatXY.shape)
x = torch.ones_like(trainX)
y = torch.ones_like(trainY)
# print(x.shape, y.shape)
print(trainY.shape)
train = torch.utils.data.TensorDataset(trainY, trainY)
loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)  # train_loader ready



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        #self.fc2 = nn.Linear(hidden_size, hidden2_size)
        # self.fc3 = nn.Linear(hidden2_size, output_size) # output # of predicted traits

    def forward(self, input):
        input = self.fc1(input)
        # input = self.activation1(input)
        # input = self.fc2(input)
        # input = self.activation1(input)
        # input = self.fc3(input)
        return input


def train(model, criterion = nn.MSELoss(),  momentum = momentum):
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss = 0
        for i, data in enumerate(loader):
            # get the inputs
            inputs, labels = data
            labels = Variable(labels)
            # inputs = Variable(inputs.view(batch_size, 1, -1), requires_grad=True)
            inputs = Variable(inputs.view(batch_size, -1))
            optimiser.zero_grad() # clear grads

            # start forward
            output = model(inputs).squeeze(1)
            l = criterion(output, labels)
            loss += l
            loss.backward(retain_graph=True) # backpropagations
            optimiser.step() # update the parameters
            optimiser.zero_grad()

        print('Epoch:  %d | Loss: %.4f' % (epoch + 1, loss.item()))

# def test(model):
#     test = hkl.load("dontPush/test.hkl")
#     test = torch.from_numpy(test).float()
#     test_var = Variable(test)
#     test_output = model(test_var)
#     test_output = test_output.data.numpy()
#     print(test_output)


def main():
    model = Net()
    training = train(model)


if __name__ == '__main__':
    main()
    # plot loss