from pysnptools.snpreader import Bed

# data for something i dont remember
start = 0
end = 99
# filename = "dontPush/test_pheno100.csv"



# variables for transformingData.py
num_snps = 10000
samples = 500

train_loop_samples = 80 # 40kX20k
test_loop_samples = 20 # 10kX20k

snpsFile = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
tot_patients = snpsFile.iid_count # 488377 tot patients
tot_snps = snpsFile.sid_count # 63487 tot snps
# snps_subset = snpsFile[:, :20000]
STARTMEMORY, STOPMEMORY = 100000, 100500 # this is nedded to save only a portion in memory

