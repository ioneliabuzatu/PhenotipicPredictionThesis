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
from dataLoaders import train_data, test_data

writer = tensorboardX.SummaryWriter()

# parameters
hidden_size = 1000
batch_size = 10
epochs = 100
learning_rate = 0.0001  # decrease lr if loss increases, or increase lr if loss decreases.


class FC(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.predict = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = F.relu(self.hidden(x))
        x= self.hidden(x)
        x = self.predict(x)
        return x



criterion = nn.SmoothL1Loss()
testLoss = nn.MSELoss()
doesitwork = nn.L1Loss()


def training(net, loader, optimizer):
    data_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
    for epoch in range(epochs):
        loss = 0
        for i, data in enumerate(data_loader):
            step = epoch * len(data_loader) + i
            inputs, labels = data

            prediction = net(inputs)

            original_prediction = prediction * y_var + y_mean
            original_label = labels * y_var + y_mean
            if epoch == epochs - 1:
                print(original_prediction, original_label)
            # # visual_loss = doesitwork(original_label, original_prediction)
            # writer.add_scalar('Train/doesitwork', visual_loss, step)

            l = criterion(prediction, labels)
            loss += l

            # backpropagation
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #        writer.add_scalar('Train/weights', net.predict.weight[-1], epoch)

            writer.add_scalar('TrainFC/Loss', l.item(), step)
            # net.log_weights(step)

        writer.add_scalar('Epoch/TrainFc/Loss', loss, epoch)
        print('Epoch:  %d | Loss: %.4f ' % (epoch + 1, loss.item()))

    return "Finished Training"





train_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
model = FC(x_features, y_features)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
training(model, train_loader, optimizer)
# torch.save(model.state_dict(), 'modelWeights/paramsFC.ckpt')
# try:
#     model.load_state_dict(torch.load('paramsFC.ckpt'))
# except FileNotFoundError:
#     train = training(model, train_loader, optimizer)
#     torch.save(model.state_dict(), 'paramsFC.ckpt')
# model.load_state_dict(torch.load('paramsFC.ckpt'))
# train = training(model, data, optimizer)

test_loader, test_x_mean, test_x_var, test_y_mean, test_y_var = test_data()

# with torch.no_grad():
#     test_loss_tot = 0
#     accuracy = 0
#
#     i = 0
#     for inp, lab in test_loader:
#         # inputs = inputs
#         # labels = inputs
#
#         predictioN = model(inp)
#
#         original_predictioN = predictioN * test_y_var + test_y_mean
#         # original_labeL = lab * y_var + y_mean
#         test_loss = testLoss(original_predictioN, lab)
#         test_loss_tot += test_loss
#         # print(original_predictioN, lab)
#         writer.add_scalar("TestFC/Loss", test_loss, i)
#         i += 1

    # print(test_loss_tot)