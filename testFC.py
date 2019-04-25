from dataLoaders import test_data
from fullyConnected import model, writer
import torch
import torch.nn as nn

test_loader, x_mean, x_var, y_mean, y_var, lenY = test_data()

msl = nn.MSELoss()


def testing_fc(net):
    with torch.no_grad():
        test_loss_tot = 0
        accuracy = 0
        i = 0
        visual_loss_tot = 0
        for inputs, labels in test_loader:
            prediction = net(inputs)

            original_prediction = prediction * y_var + y_mean
            original_label = labels * y_var + y_mean
            test_loss = msl(prediction, labels)
            test_loss_tot += test_loss * labels.size(0)  # labels.size(0)
            # * batch size
            visual_loss = msl(original_label, original_prediction)
            visual_loss_tot += visual_loss
            writer.add_scalar('Test/doesitwork', visual_loss_tot / lenY)

            # print(original_prediction, labels, i)
            # writer.add_scalar("/Step/Loss", test_loss.item(), i)
            i += 1

        print("test loss is {}".format(test_loss_tot / lenY))
        return test_loss_tot


if __name__ == "__main__":
    # double check parameters file exists
    try:
        model.load_state_dict(torch.load('modelWeights/paramsFC.ckpt'))
    except FileNotFoundError:
        print("WARNING: file 'modelWeights/paramsFC.ckpt' not found" + "\n" + "TRAINING NOW...")

    testing_fc(model)
