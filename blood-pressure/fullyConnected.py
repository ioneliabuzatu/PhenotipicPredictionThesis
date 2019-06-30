import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from dataLoaders import train_data

writer = tensorboardX.SummaryWriter()

# parameters
hidden_size = 100  # maybe 2000 is too big, try with smaller size
batch_size = 50
epochs = 100
learning_rate = 0.0001  # decrease lr if loss increases, or increase lr if loss decreases.


class FC(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.predict = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.tanh(self.predict(x))
        return x


criterion = nn.SmoothL1Loss()
msl = nn.MSELoss()
doesitwork = nn.L1Loss()


def training(net, loader, optimizer):
    data_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
    for epoch in range(epochs):
        loss = 0
        loss_denormalized = 0
        for i, data in enumerate(data_loader):
            step = epoch * len(data_loader) + i
            inputs, labels = data

            prediction = net(inputs)

            original_prediction = prediction * y_var + y_mean
            original_label = labels * y_var + y_mean
            # if epoch == epochs - 1:
            #     print(original_prediction, original_label)
            visual_loss = criterion(original_label, original_prediction)
            # writer.add_scalar('Train/doesitwork', visual_loss, step)
            loss_denormalized += visual_loss

            l = criterion(prediction, labels)
            loss += l * labels.size(0)

            # backpropagation
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #        writer.add_scalar('Train/weights', net.predict.weight[-1], epoch)

            # writer.add_scalar('/Step/Loss', l.item(), step)
            # net.log_weights(step)

        # writer.add_scalar('Epoch/TrainFc/deno', loss_denormalized, epoch)
        # writer.add_scalar('Epoch/TrainFc/Loss', loss, epoch)
        print('Epoch:  %d | Loss: %.4f ' % (epoch + 1, loss))

    return "Finished Training"


train_loader, x_features, y_features, x_mean, x_var, y_mean, y_var = train_data()
model = FC(x_features, y_features)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# if weights were not saves before run training and save parameters
try:
    model.load_state_dict(torch.load('modelWeights/paramsFC.ckpt'))
except FileNotFoundError:
    training(model, train_loader, optimizer)
    torch.save(model.state_dict(), 'modelWeights/paramsFC.ckpt')

# training(model, train_loader, optimizer)
