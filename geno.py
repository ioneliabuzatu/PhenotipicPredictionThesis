from pysnptools.snpreader import Bed, Pheno
import numpy as np
import hickle as hkl
import os
import h5py
import pickle
import argparse  # add the sizes as an argument to bash
from config import *


snps = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
snps_shape = snps.shape
tot_patients = snps_shape[0] # 488377 but subset of 100
tot_snps = snps_shape[1] # 63487


def getX(test_subset = subset, num_snps = number_of_snps, snps = snpsFile):
    snps  = snps[400:, :50000]
    makeFlatGeno = [[] for i in range(test_subset)]

    for patient_id in range(0, test_subset):
        subset = snps[patient_id, :num_snps]  # get un array af all snps for each patient
        snp_val = subset.read().val
        where_are_nan = np.isnan(snp_val)
        snp_val[where_are_nan] = -1
        i = 0
        for snp in snp_val[0]:
            if snp == 0:
                makeFlatGeno[patient_id].extend([1, 0, 0])
            elif snp == 1:
                makeFlatGeno[patient_id].extend([1, 1, 0])
            elif snp == 2:
                makeFlatGeno[patient_id].extend([1, 0, 1])
            else:
                makeFlatGeno[patient_id].extend([1, 0, 0])
            i += 1

    makeFlatGenoNumpy=np.array(makeFlatGeno)

    return makeFlatGenoNumpy


testOutput = getX()
hkl.dump(testOutput, 'dontPush/geno10000X20000.hkl', mode='w')

