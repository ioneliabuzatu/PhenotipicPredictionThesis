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



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # (in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(23, 18, (2, 1))

        #(kernel_size, stride, padding) – applies max pooling
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(18, 32, (2,1))
        self.fc1 = nn.Linear(32 * 5998, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # xize changes from (1,20,100) to (50,20,100)
        x = torch.relu_((self.conv1(x.unsqueeze(3))))
        x = self.pool(x)
        x = torch.relu_(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu_(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


