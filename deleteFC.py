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


# parameters
output_size = 1 # predict one trait for now
input_size = 200 # tot snps or observations for sample
hidden_size = 500
batch_size = 200
obsevations_at_step1 = 1 # one at a time then change it to 300 --> corresponds to seq_len
num_layers = 1 # two layers rnn


# hyperparamters
epochs = 200
learning_rate = 0.001  # decrease lr if loss increases, or increase lr if loss decreases.


# data loading: train and test
trainX = hkl.load('dontPush/geno200X500.hkl')
trainX = torch.from_numpy(trainX).float()
trainY = pd.read_csv('dontPush/pheno200X500.csv', sep="\t")
trainY = torch.tensor(trainY["f.4079.0.0"].values).float()

#### for debuggin
concatXY = np.column_stack((trainX, trainY))
concatXY = torch.from_numpy(concatXY)
# print(concatXY.shape)
x = torch.ones_like(trainX)
y = torch.ones_like(trainY)
# print(x.shape, y.shape)
print(trainY.shape)
train = torch.utils.data.TensorDataset(trainY, trainY)
loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 50, 1, 200, 1

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 0.05
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(trainY)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, trainY)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


print("###########################")
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(loader):
        inputs, labels = data

        prediction = model(inputs)

        l = loss_fn(prediction, labels)
        loss += l
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Apply gradients
        for param in model.parameters():
            param.data.add_(-0.1 * param.grad.data)

    print('Epoch:  %d | Loss: %.4f' % (epoch + 1, loss.item()))