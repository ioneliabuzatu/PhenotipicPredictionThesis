import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import hickle as hkl
import torchvision.transforms as transforms
import tensorboardX
from dataLoaders import train_data, test_data, train_val_test
from torch.utils import data

writer = tensorboardX.SummaryWriter()


# 3000 * 6000
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # (in_channels, out_channels, kernel_size, stride, padding)
        self.kernel_size = 3
        self.paddings = self.kernel_size // 2

        self.conv1 = nn.Conv1d(1, 4, kernel_size=self.kernel_size, stride=5, padding=self.paddings)
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=3, padding=self.paddings)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=self.kernel_size, stride=3, padding=self.paddings)
        self.fc1 = nn.Linear(2144 * 1, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        # print("oroginal x " + str(x.size()))
        # xize changes from (1,20,100) to (50,20,100)
        # print("no unsqueeze: " + str(x.unsqueeze(1).size()))
        x = torch.relu_(self.conv1(x.unsqueeze(1)))  # unsqueeze 1 not 2
        # print("after unsq " + str(x.size()))
        x = self.pool(x)
        # print("after pool " + str(x.size()))
        x = torch.relu_(self.conv2(x))
        # x = self.pool(x)
        # print("after 2nd conv " + str(x.size()))
        x = x.view(x.size(0), -1)
        # print("after x.view(x.size(0), -1) " + str(x.size()))
        x = torch.relu_(self.fc1(x))
        # print("after fc1 " + str(x.size()))
        x = torch.tanh(self.fc2(x))
        # print("final " + str(x.size()))
        # print("done")
        return x


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

net = CNN()

epochs = 100
learning_rate = 0.0001

# train_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
# train, validation and test sets with length
# train_set, val_set, test_set = data.random_split(train_loader, (2100, 900, 0))


train_loader, validation_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_val_test()

# alternative
from sklearn.model_selection import train_test_split

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Loss functions
# loss = nn.CrossEntropyLoss()
criterion = nn.SmoothL1Loss()
mean = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_loss = 0  # sum all the losses and divide by the number of batches
    deno_loss = 0
    for i, data in enumerate(train_loader):
        train_step = epoch * len(train_loader) + i
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        # inputs = inputs.view(inputs.size(0), -1)
        prediction = net(inputs)

        original_prediction = prediction * y_var + y_mean
        original_label = labels * y_var + y_mean
        # if epoch == epochs - 1:
        #     print(original_prediction, original_label)
        visual_loss = criterion(original_label, original_prediction)
        deno_loss += visual_loss

        l = criterion(prediction, labels)
        train_loss += l

        # backpropagation
        l.backward()
        optimizer.step()
        writer.add_scalar('/Step/Loss', l.item(), train_step)

    # low variance if the gap between the train and validation loss is really close
    net.eval()
    val_loss = 0
    for i, data in enumerate(validation_loader):
        step_val = epoch * len(train_loader) + i
        input, label = data
        output = net(input)

        loss_validation = criterion(output, label)
        val_loss += loss_validation

        writer.add_scalar('/Step/Loss', loss_validation.item(), step_val)

    # writer.add_scalar('Epoch/TrainFc/deno', deno_loss, epoch)
    writer.add_scalar('Epoch/Loss', train_loss, epoch)
    writer.add_scalar('Epoch/Loss', val_loss, epoch)
    print('Epoch:  %d | Train Loss: %.4f  | Val Loss: %.4f' % (epoch + 1, train_loss, val_loss))


if __name__ == "__main__":
    pass

"""

try:
    net.load_state_dict(torch.load('modelWeights/paramsCNN.ckpt'))
except FileNotFoundError:
    # training(model, train_loader, optimizer)
    torch.save(net.state_dict(), 'modelweights/paramsCNN.ckpt')


"""
