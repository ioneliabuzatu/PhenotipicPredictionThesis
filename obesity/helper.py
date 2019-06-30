"""Load data and others for training the obesity classfier"""

import pandas as pd
import torch
from torch.utils import data
import numpy as np
import argparse
import hickle as hkl
import os
import shutil


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Remove not existing patients in calls')

    parser.add_argument('--genos', dest='genos',
                        help='loading patients genotypes',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_genos_100.hkl',
                        type=str)

    parser.add_argument('--phenos', dest='phenos',
                        help='loading patients pehnotypes with labels',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_labels.csv',
                        type=str)

    parser.add_argument('--val_genos', dest='val_genos',
                        help='loading patients genotypes',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_val_25.hkl',
                        type=str)

    parser.add_argument('--val_phenos', dest='val_phenos',
                        help='loading patients pehnotypes with labels',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_labels.csv',
                        type=str)



    parser.add_argument('--snps', dest='snps',
                        help='bed fam files prefix',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/dontPush/calls/ukb_cal_chr1_v2', type=str)

    parser.add_argument('--to_dir', dest='to_dir',
                        help='path and name where to seve cleaned csv',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/cleaned.csv',
                        type=str)

    parser.add_argument('--bmiy', dest='bmiy',
                        help='cleaned csv to add labels column',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/cleaned.csv',
                        type=str)

    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='training loops',
                        default=20,
                        type=int)

    parser.add_argument('--criterion', dest='criterion',
                        help='nn.MSELoss()',
                        default='nn.MSELoss()',
                        type=str)

    parser.add_argument('--optimizer', dest='optimizer',
                        help='nn.MSELoss()',
                        default='adam',
                        type=str)

    args = parser.parse_args()
    return args


args = parse_args()


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(data)
        y = self.labels[ID]

        return X, y


# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}


def train_data():
    trainX = hkl.load(args.genos)
    trainX = torch.from_numpy(trainX).float()
    trainY = pd.read_csv(args.phenos, sep=" ")
    trainY = torch.tensor(trainY["num_labels"][:3].values).long().unsqueeze(-1)

    # y_mean = trainY.mean()
    # y_var = trainY.var()
    concatXY = np.column_stack((trainX, trainY))
    concatXY = torch.from_numpy(concatXY)
    # x_mean = trainX.mean(dim=0)
    # x_var = trainX.var(dim=0)

    # normalization for x and y
    # trainX -= trainX.mean(dim=0)
    # trainX /= trainX.var(dim=0) + 1e-10  # a small number is added to prevent nan when sd == 0
    # trainY -= trainY.mean()
    # trainY /= trainY.var()
    # concatXY -= concatXY.mean(dim=0)
    # concatXY /= concatXY.var(dim=0)

    train = torch.utils.data.TensorDataset(trainX, trainY)
    # TODO: think of a way to import only once batch_size evrywhere
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, trainX.shape[1], trainY.shape[1]  # , x_mean, x_var, y_mean, y_var, len(trainY)


# loader, sx, sy = train_data()
# print(sx, sy)


#
# for x, y in loader:
#     print(x,y)


def make_classes(phenos):
    for row in range(486382):

        if phenos['f.21001.0.0'][row] < 18.5:
            phenos.at[row, 'labels'] = 'Underweight'

        elif phenos['f.21001.0.0'][row] >= 18.5 and phenos['f.21001.0.0'][row] <= 24.9:
            phenos.at[row, 'labels'] = 'Normal'


        elif phenos['f.21001.0.0'][row] >= 25 and phenos['f.21001.0.0'][row] <= 29.9:
            phenos.at[row, 'labels'] = 'Overweight'

        else:
            phenos.at[row, 'labels'] = 'Obese'
    pass


def save_checkpoint(state, output_dir):
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))
    filepath = os.path.join(output_dir, 'epoch_{:03}.pth.tar'.format(state['epoch']))
    torch.save(state, filepath)
    shutil.copyfile(filepath, os.path.join(output_dir, 'last_checkpoint.pth.tar'))


def load_checkpoint(checkpoint_path, model, optimizer):
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
