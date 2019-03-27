from pysnptools.snpreader import Bed

# data for something i dont remember
start = 0
end = 99
# filename = "dontPush/test_pheno100.csv"


# variables for transformingData.py
num_snps = 5000
samples = 2500
snpsFile = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
tot_patients = snpsFile.iid_count # 488377 tot patients
tot_snps = snpsFile.sid_count # 63487 tot snps
snps_subset = snpsFile[:, :20000]

