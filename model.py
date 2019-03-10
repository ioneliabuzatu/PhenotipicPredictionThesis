import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

# hyperparameters
input_size = 300
hidden_size = 100
hidden2_size = 50
output_size = 1 # TODO: how many?

epochs = 7
batch_size = 20
learning_rate = 0.005
momentum = 0
loss_func = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

# make data to tensor
xTrainData = hkl.load('dontPush/trainGeno.hkl')
yTrainData = pd.read_csv('dontPush/pheno100.csv', sep="\t")

xTrainData = xTrainData.transpose([1,0,2]).reshape(100,300)
# print(xTrainData.shape)
xTrainData  = torch.from_numpy(xTrainData).float()

yTrainData = torch.tensor(yTrainData["BlodP"].values).float()
print("the mean of train is {}".format(torch.mean(yTrainData)))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size) # output # of predicted traits
        self.activation1 = nn.Tanh()

        # TODO: return classifier fro traits
    def forward(self, input):
        input = self.fc1(input)
        input = self.activation1(input)
        input = self.fc2(input)
        input = self.activation1(input)
        input = self.fc3(input)
        return input

model = Net()
# print(model)

# def train(model, criterion = nn.MSELoss(), optimizer = optim.SGD(model.parameters(), lr = learning_rate), momentum = momentum):


# loss = criterion(output, target)
optimiser = optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    for i in range(0, xTrainData.shape[0], batch_size):
        x_mini = xTrainData[i:i + batch_size]
        y_mini = yTrainData[i:i + batch_size]

        x_var = Variable(x_mini)
        y_var = Variable(y_mini)
    # inputs = Variable(xTrainData)
    # target = Variable(yTrainData)

        optimiser.zero_grad() # clear grads

    # start forward
        output = model(x_var)
        loss = criterion(output, y_var)
        loss.backward() # backpropagations
        optimiser.step() # update the parametrs
        print("epoch {}, loss {}".format(epoch, loss.item()))


test = hkl.load("dontPush/test.hkl")
test = torch.from_numpy(test).float()
test_var = Variable(test)
test_output = model(test_var)
test_output = test_output.data.numpy()
print(test_output)
print("the mean of test is {}".format(np.mean(test_output)))