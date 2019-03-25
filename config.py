from pysnptools.snpreader import Bed


start = 0
end = 99
filename = "dontPush/pheno100.csv"

num_snps = 5

samples = 20
snpsFile = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)
snps_subset = snpsFile[400:, :20000]