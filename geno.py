from pysnptools.snpreader import Bed, Pheno
import numpy as np
import hickle as hkl
import os
import h5py

snps = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
snps_shape = snps.shape
tot_patients = snps_shape[0] # 488377 but subset of 100

tot_snps = snps_shape[1] # 63487


def getgeno(train_subset = 100, num_snps = 100):
    train_geno_data = np.zeros(shape=(train_subset, num_snps, 3))
    for patient_id in range(0, train_subset):
        subset = snps[patient_id, :num_snps]  # get un array af all snps for each patient
        snp_val = subset.read().val
        where_are_nan = np.isnan(snp_val)
        snp_val[where_are_nan] = -1

        # prepare the input
        i = 0
        for snp in snp_val[0]:
            if snp == 0:
                train_geno_data[patient_id][i] = [1, 0, 0]
            elif snp == 1:
                train_geno_data[patient_id][i] = [0, 1, 0]
            elif snp == 2:
                train_geno_data[patient_id][i] = [0, 0, 1]
            else:
                train_geno_data[patient_id][i] = [0, 0, 0]
            i += 1
    return train_geno_data


outputGetgeno = getgeno()
# hkl.dump(train_geno_data, 'data/trainGeno.hkl', mode='w')


# make flat vector
def toflatgeno(train_subset = 100, num_snps = 100):
    makeFlatGeno = [[] for i in range(train_subset)]
    for patient_id in range(0, train_subset):
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
    return makeFlatGeno


outputFlat = toflatgeno()
# hkl.dump(outputFlat, 'data/trainFlatGeno.hkl', mode='w')