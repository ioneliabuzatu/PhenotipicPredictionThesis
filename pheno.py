from pysnptools.snpreader import Bed, Pheno
import numpy as np
import hickle as hkl
import os
import h5py

snps = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)


def getpatientid(bedfile):
    pheno_patient_id = []
    for patient in snps.iid[:100]:
        id = patient[0].decode('utf8').strip()  # from 2 to -1
        id = int(id)
        pheno_patient_id.append(id)

    return pheno_patient_id

# store id patient to get pheno
output_id = getpatientid(snps)


def maketxtid(array):
    with open("patient_ids.txt", "w") as f:
        for item in array:
            f.write("%s\n" % item)

    return "Done"


