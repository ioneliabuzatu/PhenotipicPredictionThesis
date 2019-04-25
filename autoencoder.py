import torch
import torch.nn as nn
import torch.nn.functional as F
from dataLoaders import train_data
import tensorboardX
import torch.optim as optim

writer = tensorboardX.SummaryWriter()

data_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
epochs = 100
learning_rate = 0.0001
criterion = nn.SmoothL1Loss()
testLoss = nn.MSELoss()
doesitwork = nn.L1Loss()


class VAE(nn.Module):
    def __init__(self, input_size, output_size):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_size, 6000),
                                     nn.Tanh(),
                                     nn.Linear(6000, 1000),
                                     nn.Tanh(),
                                     nn.Linear(1000, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 10)
                                     )
        self.decoder = nn.Sequential(nn.Linear(10, 100),
                                     nn.Tanh(),
                                     nn.Linear(100, 50),
                                     nn.Tanh(),
                                     nn.Linear(50, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, output_size),
                                     nn.Tanh()
                                     )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


net = VAE(x_features, y_features)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(epochs):
    loss = 0
    for i, data in enumerate(data_loader):
        step = epoch * len(data_loader) + i
        inputs, labels = data

        prediction = net(inputs)

        original_prediction = prediction * y_var + y_mean
        original_label = labels * y_var + y_mean
        # if epoch == epochs - 1:
        #     print(original_prediction, original_label)
        # # visual_loss = doesitwork(original_label, original_prediction)
        # writer.add_scalar('Train/doesitwork', visual_loss, step)

        l = criterion(prediction, labels)
        loss += l

        # backpropagation
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        #        writer.add_scalar('Train/weights', net.predict.weight[-1], epoch)

        writer.add_scalar('/Step/Loss', l.item(), step)
        # net.log_weights(step)

    writer.add_scalar('Epoch/TrainFc/Loss', loss, epoch)
    print('Epoch:  %d | Loss: %.4f ' % (epoch + 1, loss.item()))
