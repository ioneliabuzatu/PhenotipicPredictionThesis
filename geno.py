from pysnptools.snpreader import Bed, Pheno
import numpy as np
import hickle as hkl
import os
import h5py
import pickle
import argparse  # add the sizes as an argument to bash
from config import *
import tqdm

snps = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
snps_shape = snps.shape
tot_patients = snps_shape[0] # 488377 but subset of 100
tot_snps = snps_shape[1] # 63487

# data with 0 and 1 and na = 0 0 0
# def getX(subset = subset, num_snps = number_of_snps, snps = snpsFile):
#     snps  = snps[400:, :50000]
#     makeFlatGeno = [[] for i in range(subset)]
#
#     for patient_id in tqdm.tqdm(range(subset), total=subset):
#         subset = snps[patient_id, :num_snps]  # get un array af all snps for each patient
#         snp_val = subset.read().val
#         where_are_nan = np.isnan(snp_val)
#         snp_val[where_are_nan] = -1
#         i = 0
#         for snp in snp_val[0]:
#             if snp == 0:
#                 makeFlatGeno[patient_id].extend([1, 0, 0])
#             elif snp == 1:
#                 makeFlatGeno[patient_id].extend([0, 1, 0])
#             elif snp == 2:
#                 makeFlatGeno[patient_id].extend([0, 0, 1])
#             else:
#                 makeFlatGeno[patient_id].extend([0, 0, 0])
#             i += 1
#         print(patient_id)
#     makeFlatGenoNumpy=np.array(makeFlatGeno)
#
#     return makeFlatGenoNumpy
#
#
# testOutput = getX()
# hkl.dump(testOutput, 'dontPush/geno5000X10000.hkl', mode='w')


def getX(num_samples=samples, num_snps=num_snps, snps=snps_subset):
    print('allocating memory')
    dataset = np.zeros(shape=(num_samples, num_snps * 3))
    print('done')

    for patient_id in tqdm.tqdm(range(num_samples)):
        subset_one = snps[patient_id, :num_snps] # one patient at a time
        snp_val = subset_one.read(order='A', view_ok=True).val # gets the genotype
        offset = 0
        for snp in snp_val[0]: # snp_val it's a 2 dimentional np array so 0 is needed
            if not np.isnan(snp): # zero by default, but insert 1 where the geno is
                assert -1 < snp < 3 # only 0, 1, and 2 or minor homozygous, heterozygous and major homozygous
                dataset[patient_id][int(snp + offset)] = 1
            offset += 3

    return dataset


out = getX()
out_shape = out.shape
print(out.shape)
# hkl.dump(out, 'dontPush/geno200X500.hkl', mode='w')

# def getXoriginal(num_samples=samples, num_snps=num_snps, snps=snps_subset):
#     print('allocating memory')
#     dataset = np.zeros(shape=(num_samples, num_snps))
#     print('done')
#
#     for patient_id in tqdm.tqdm(range(num_samples)):
#         subset_one = snps[patient_id, :num_snps] # one patient at a time
#         snp_val = subset_one.read(order='A', view_ok=True).val # gets the genotype
#         offset = 0
#         for snp in snp_val[0]: # snp_val it's a 2 dimentional np array so 0 is needed
#
#             if not np.isnan(snp): # zero by default, but insert 1 where the geno is
#                 assert -1 < snp < 3 # only 0, 1, and 2 or minor homozygous, heterozygous and major homozygous
#                 dataset[patient_id][offset] = snp
#             else:
#                 dataset[patient_id][offset] = 5 # substitute nan's by 5
#             offset += 1
#         print(patient_id)
#
#     return dataset
#
#
# original = getXoriginal()
# # print(original)
# hkl.dump(original, 'dontPush/geno200X500.hkl', mode='w')
