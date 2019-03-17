from pysnptools.snpreader import Bed


start = 0
end = 99
filename = "dontPush/pheno100.csv"

number_of_snps = 20000

subset = 100000
snpsFile = Bed('nogitdata/ukb_cal_chr' + str(1) + '_v2',count_A1=False)