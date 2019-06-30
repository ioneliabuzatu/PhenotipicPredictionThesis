"""Testing RNN"""

import torch
from dataLoaders import test_data

test_loader, x_mean, x_var, y_mean, y_var = test_data()

with torch.no_grad():
    test_loss_tot = 0
    accuracy = 0
    i = 0
    for id, pressure in test_loader:

        id = id.view(batch_size, 1, -1)

        test_prediction = net(id)

        original_prediction = test_prediction * y_var + y_mean
        original_label = pressure * y_var + y_mean

        test_loss = criterion(original_prediction, pressure)
        test_loss_tot += test_loss

        # plot test loss
        writer.add_scalar('Test/Loss', test_loss, i)

        i+=1 # needed for plotting loss at each step

        print(original_prediction, pressure)

    print("test loss is {}".format(test_loss_tot))