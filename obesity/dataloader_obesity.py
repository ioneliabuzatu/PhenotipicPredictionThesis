import numpy as np
import hickle as hkl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd


class ObesityDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(ObesityDataloader, self).__init__(*args, **kwargs)


def getLabelValue(label):
    if label == "Underweight":
        return torch.LongTensor(np.array([0], dtype=np.int64))
    elif label == "Normal":
        return torch.LongTensor(np.array([0], dtype=np.int64))
    elif label == "Overweight":
        return torch.LongTensor(np.array([0], dtype=np.int64))
    else:
        return torch.LongTensor(np.array([1], dtype=np.int64))


class ObesityDataset:

    def __init__(self, genos_path, phenos, transform=None):
        self.genos_path = genos_path
        self.phenos = phenos
        self.labels = []
        label_file = self.phenos
        for row in label_file.iterrows():
            self.labels.append(row[1][3])
        self.transform = transform

    def __getitem__(self, index):
        geno = hkl.load(self.genos_path)
        geno = torch.from_numpy(geno).float()
        geno = geno[index]
        qn = torch.norm(geno, p=2, dim=0).detach()
        geno = geno.div(qn.expand_as(geno))
        # if self.transform is not None:
        #     img = self.transform(img)
        label = getLabelValue(self.labels[index])
        return geno, label

    def __len__(self):
        return len(self.labels)
