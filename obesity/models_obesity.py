import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class CNN_BMI(nn.Module):

    def __init__(self):
        super(CNN_BMI, self).__init__()

        # (in_channels, out_channels, kernel_size, stride, padding)
        self.kernel_size = 3
        self.paddings = self.kernel_size // 2
        self.stride = 3
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv1d(1, 4, kernel_size=self.kernel_size, stride=5, padding=self.paddings)
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=3, padding=self.paddings)
        # self.conv2 = nn.Conv1d(4, 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.paddings)

        self.fc1 = nn.Linear(50784, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        # print("oroginal x " + str(x.size()))
        # xize changes from (1,20,100) to (50,20,100)
        # print("no unsqueeze: " + str(x.unsqueeze(1).size()))
        x = torch.relu(self.conv1(x.unsqueeze(1)))  # unsqueeze 1 not 2

        # x = self.pool(x)
        # print("after 2nd conv " + str(x.size()))
        x = x.view(x.size(0), -1)
        # print("after x.view(x.size(0), -1) " + str(x.size()))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # print("after fc1 " + str(x.size()))
        x = torch.sigmoid(self.fc2(x))
        # print("final " + str(x.size()))
        # print("done")
        return x


class FC(nn.Module):

    def __init__(self, input_size, output_size):
        super(FC, self).__init__()
        self.dropout = nn.Dropout(p=0.50)
        self.hidden_size = 1000

        self.l1 = nn.Linear(input_size, self.hidden_size)
        # self.l = nn.Linear(self.hidden_size, 100)
        self.l2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.dropout(x)
        # x = torch.relu(self.l(x))
        # x = self.dropout(x)
        x = torch.sigmoid(self.l2(x))
        return x
