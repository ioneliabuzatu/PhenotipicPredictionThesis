import os
import datetime
import tqdm
import pandas as pd
from models_obesity import FC, CNN_BMI
from helper import parse_args
import torch
import torch.nn as nn
from dataloader_obesity import ObesityDataset, ObesityDataloader
import torchvision.transforms as transforms
import tensorboardX
import torch.nn.functional as F

print("COMMENT PLOT!")
comment = input()
tensorboard_dir = os.path.join("runs", "{}:{}".format(datetime.datetime.now().strftime('%b%d_%H-%M-%S'), comment))
writer = tensorboardX.SummaryWriter(tensorboard_dir, flush_secs=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration

args = parse_args()

# training_generator, shape_x, shape_y = train_data()
# net = FC(63480, 1)  # binary classfier
# print(net)

net = CNN_BMI()
print(net)

optimizer = None

if args.optimizer == "adam":
    # lr = lr * 0.1
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

criterion1 = nn.NLLLoss()  # for quick and small tasks
criterion2 = nn.CrossEntropyLoss()  # considers both right and wrong
criterion3 = nn.BCELoss()
criterion4 = nn.BCEWithLogitsLoss()

# training data
TRAIN_GENOS_PATH = args.genos
TRAIN_LABELS_PATH = args.phenos
transformations = transforms.Compose([transforms.ToTensor()])
train_dataset = ObesityDataset(TRAIN_GENOS_PATH, pd.read_csv(TRAIN_LABELS_PATH, sep=' ', header=0)[:100],
                               transformations)
train_loader = ObesityDataloader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

# validation data
VAL_GENOS_PATH = args.val_genos
VAL_LABELS_PATH = args.val_phenos
transformations = transforms.Compose([transforms.ToTensor()])
val_dataset = ObesityDataset(VAL_GENOS_PATH, pd.read_csv(VAL_LABELS_PATH, sep=' ', header=0)[-25:], transformations)
val_loader = ObesityDataloader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# test data
# TRAIN_GENOS_PATH = args.genos
# TRAIN_LABELS_PATH = args.phenos
# transformations = transforms.Compose([transforms.ToTensor()])
# train_dataset = ObesityDataset(TRAIN_LABELS_PATH, TRAIN_GENOS_PATH, transformations)
# train_loader = ObesityDataloader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

# Loop over epochs
for epoch in tqdm.tqdm(range(args.max_epochs)):
    net.train()
    loss_temp = 0
    # Training
    for i, (local_batch, local_labels) in enumerate(train_loader):

        step_train = epoch * len(train_loader) + i

        # Transfer to GPU
        batch, labels = local_batch.to(device), local_labels.to(device)
        local_labels = local_labels.float()
        # Model computations
        net.zero_grad()
        prediction = net(local_batch)

        # local_labels = torch.squeeze(local_labels, 1)

        # if local_labels.item() == 1:
        #     local_labels = torch.tensor([[0, 1]]).float()
        # else:
        #     local_labels = torch.tensor([[1, 0]]).float()

        # print(prediction, local_labels)
        loss = criterion3(prediction, local_labels)
        loss_temp += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('/Step/Train', loss.item(), step_train)
    writer.add_scalar('Epoch/Train', loss_temp, epoch)

    val_loss = 0
    for i, (val_batch, val_labels) in enumerate(val_loader):
        step_val = epoch * len(train_loader) + i
        val_labels = val_labels.float()
        # one-hot encoding
        # if val_labels.item() == 1:
        #     val_labels = torch.tensor([[0, 1]]).float()
        # else:
        #     val_labels = torch.tensor([[1, 0]]).float()

        output = net(val_batch)

        loss_validation = criterion3(output, val_labels)
        val_loss += loss_validation

        writer.add_scalar('/Step/Val', loss_validation.item(), step_val)
    #
    writer.add_scalar('Epoch/Val', val_loss, epoch)
    # print('\tEpoch:  %d | Train Loss: %.4f ' % (epoch + 1, loss_temp))
    print('\tEpoch:  %d | Train Loss: %.4f | Val Loss: %.4f ' % (epoch + 1, loss_temp, val_loss))
