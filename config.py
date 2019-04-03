from pysnptools.snpreader import Bed

# data for something i dont remember
start = 0
end = 99
# filename = "dontPush/test_pheno100.csv"


# variables for transformingData.py
samples = 500
num_snps = 6000
num_samples = 3000

train_loop_samples = 2 # 40kX20k
test_loop_samples = 20 # 10kX20k

snpsreader = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
tot_patients = snpsreader.iid_count # 488377 tot patients
tot_snps = snpsreader.sid_count # 63487 tot snps
# snps_subset = snpsFile[:, :20000]
STARTMEMORY, STOPMEMORY = 100000, 100500 # this is nedded to save only a portion in memory



###################################################
#####train and test files for dataLoaders##########
###################################################
train_geno = "dontPush/3kX6k.hkl"
train_pheno =  "dontPush/pheno_3kX6k_012.csv"

test_geno = "dontPush/test_3kX6k.hkl"
test_pheno = "dontPush/phenoTEST_1kX6k_012.csv"
