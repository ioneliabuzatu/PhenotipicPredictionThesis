"""
This is for ordering patients id based on their order in genotypes table with snpreader
"""

import pandas as pd
from pysnptools.snpreader import Bed
import glob


snps = Bed('/Users/ioneliabuzatu/PycharmProjects/biobank/nogitdata/ukb_cal_chr1_v2',count_A1=False)
# the exact order from genotype table
idOrder = [] # samples id to be retrived
for patient in snps.iid[400:600]:
    id = patient[0].decode('utf8').strip()  # for binary stored indexes
    id = int(id)
    idOrder.append(id)


data = pd.read_csv('regression/full_phenotype_may', sep=" ")
# create df with header to store the final file
final_df = data.loc[data["f.eid"] == idOrder[0]]

for row in idOrder[1:]:
    new_row = data.loc[data["f.eid"] == row]
    new_row = pd.DataFrame(new_row)
    final_df = final_df.append(new_row, ignore_index=True)


final_df = final_df.fillna(final_df.mean())
final_df = final_df.iloc[:, 1:2] # unlike loc, iloc keeps the header
filename = "dontPush/pheno200X500.csv"
fileISpresent = glob.glob(filename)


if not fileISpresent:
    try:
        final_df.to_csv(filename, sep='\t', index=False)
    except FileExistsError:
        print("WARNING: this file already exists!")