from os import listdir
from pathlib import Path
import glob
from pysnptools.snpreader import Bed
import numpy as np
import hickle as hkl

# the idea is to take all the chromosomes
# np.stack([a,b], 1) to stack each iid to their chromosomals snps respectivly
# sue the matrix as input for the convolutional netza

totdataset = np.zeros(shape=(100, 5))
dataset = np.empty(shape=(100, 5))
snpsreader = Bed("nogitdata/ukb_cal_chr1_v2", count_A1=False)  # for each snp
subset = snpsreader[:100, :5]
ondisk = subset.read(order='A', view_ok=True).val
for patient_id in range(100):
    offset = 0
    for snp in ondisk[patient_id]:  # snp_val it's a 2 dimentional np array so 0 is needed
        if not np.isnan(snp):  # zero by default, but insert 1 where the geno is
            assert -1 < snp < 3  # only 0, 1, and 2 or minor homozygous, heterozygous and major homozygous
            dataset[patient_id][offset] = snp
        else:
            dataset[patient_id][offset] = 5  # substitute nan's by 5
        offset += 1

totdataset = np.stack([totdataset, dataset], 1)

print("making the file...")
hkl.dump(dataset, 'dontPush/DELETE.hkl', mode='w')
# df.to_pickle('123.pkl')
