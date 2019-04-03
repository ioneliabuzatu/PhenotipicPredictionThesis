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
from pysnptools.snpreader import Bed, Pheno
import tqdm
from config import snpsreader, samples, num_snps, train_loop_samples, test_loop_samples


def get_original(snps=snpsreader):  # 3000 x 60000
    num_samples = 500
    num_snps = 6000
    print('allocating memory')
    # first this one and 5 is the known  beforehand numbers colunmsn
    totdataset012 = np.array([], dtype=np.int64).reshape(0, 6000)
    dataset = np.zeros(shape=(num_samples, num_snps))  # then always use this one
    print('done')
    diskmemorySTART = 100000
    diskmemorySTOP = 100500

    for sample in tqdm.tqdm(range(train_loop_samples)):
        subset = snps[diskmemorySTART:diskmemorySTOP, :num_snps]
        subset_val = subset.read(order='A', view_ok=True).val  # save geneotypes to memory one loop of samples
        for patient_id in range(num_samples):
            offset = 0
            for snp in subset_val[patient_id]:  # snp_val it's a 2 dimentional np array so 0 is needed
                if not np.isnan(snp):  # zero by default, but insert 1 where the geno is
                    assert -1 < snp < 3  # only 0, 1, and 2 or minor homozygous, heterozygous and major homozygous
                    dataset[patient_id][offset] = snp
                else:
                    dataset[patient_id][offset] = 5  # substitute nan's by 5
                offset += 1
        diskmemorySTART += 500
        diskmemorySTOP += 500
        totdataset012 = np.vstack([totdataset012, dataset])

    return totdataset012


"""
def get_01(snps = snpsFile):
    num_samples = 250
    num_snps = 60000
    tot_dataset = np.array([], dtype=np.int64).reshape(0,
                                                       60000 * 3)  # first this one and 5 is the known  beforehand numbers colunmsn
    dataset = np.zeros(shape=(num_samples, num_snps * 3))  # then always use this one
    diskmemory = 0
    loop_samples = 0
    while loop_samples <= 2:
        for patient_id in tqdm.tqdm(range(num_samples)):
            subset = snps[diskmemory:, 60000]
            snp_val = subset.read(order='A', view_ok=True).val  # gets the genotype
            offset = 0
            for snp in snp_val[0]:  # snp_val it's a 2 dimentional np array so 0 is needed
                if not np.isnan(snp):  # zero by default, but insert 1 where the geno is
                    assert -1 < snp < 3  # only 0, 1, and 2 or minor homozygous, heterozygous and major homozygous
                    dataset[patient_id][int(snp + offset)] = 1
                offset += 3
        tot_dataset = np.vstack([tot_dataset, dataset])
        dataset = np.zeros(shape=(num_samples, num_snps * 3))
        loop_samples += 1
        diskmemory += 100

    return tot_dataset
"""
