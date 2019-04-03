import os
import sys
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
from dataLoaders import train_data
import datetime
import tempfile

train_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()

if '_pydev_bundle.pydev_log' in sys.modules.keys(): # if it was ran in the debugger
    tensorboard_dir = tempfile.mktemp() # create a temporary directory avoiding to put useless log folders in the mix
else:
    print("COMMENT PLOT!")
    comment = input()
    tensorboard_dir = os.path.join("runs", "{}:{}".format(datetime.datetime.now().strftime('%b%d_%H-%M-%S'), comment))


writer = tensorboardX.SummaryWriter(tensorboard_dir, flush_secs=2)

# hyperparameters
hidden_size = 100
batch_size = 1
epochs = 100
learning_rate = 0.1
hidden_size2 = 1

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, tot_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # num of hidden dim
        self.tot_layers = tot_layers  # num of hidden layers
        self.lstm = nn.LSTM(hidden_size, output_dim, batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size2, output_dim)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))
        h1 = Variable(torch.zeros(self.tot_layers, input.size(0), self.hidden_size))

        # processing feed-forward layers first improves learning
        first_fc = self.fc1(input)
        h0, h1 = self.lstm(first_fc, (h0, h1))

        return h0


model = RNN(x_features, y_features)
criterion = nn.SmoothL1Loss()
doesitwork = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(train_loader):
        step = epoch * len(train_loader) + 1
        inputs, labels = data

        inputs = Variable(inputs.view(batch_size, 1, -1))
        labels = Variable(labels)

        prediction = model(inputs)

        original_prediction = prediction * y_var + y_mean
        original_label = labels * y_var + y_mean
        # if epoch == epochs -1:
        #     print(original_prediction, original_label)

        visual_loss = doesitwork(original_label, original_prediction)
        writer.add_scalar('RNN/doesitwork', visual_loss, step)

        l = criterion(prediction, labels)
        loss += l

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        writer.add_scalar('RNN/step/l', l.item(), step)

    writer.add_scalar('RNN/Epoch/Loss', loss, epoch)
    print('Epoch:  %d | Loss: %.4f' % (epoch + 1, loss))

# try:
#     model.load_state_dict(torch.load('modelWeights/paramsRNN.ckpt'))
# except FileNotFoundError:
#     training(model, train_loader, optimizer)
#     torch.save(model.state_dict(), 'modelWeights/paramsRNN.ckpt')
#
