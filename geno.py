from pysnptools.snpreader import Bed, Pheno
import numpy as np
import hickle as hkl
import os

snps = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
snps_shape = snps.shape
tot_patients = snps_shape[0] # 488377 but subset of 1000
tot_snps = snps_shape[1] # 63487 # 10000

inputX = np.empty((0,100), int)

train_subset = 10
# first 10 patients
# 1505458
# 1738124
# 5547326
# 4351246
# 3590647
# 3714710
# 1079689
# 2753720
# 5624950
# 5016326
for patiant_id in range(0, train_subset):
    subset = snps[patiant_id, :100] # get un array af all snps for each patient
    snp_val = subset.read().val
    where_are_nan = np.isnan(snp_val)
    snp_val[where_are_nan] = -1
    inputX = np.append(inputX, snp_val, axis=0)


hkl.dump(inputX, 'trainGeno.hkl', mode='w')




