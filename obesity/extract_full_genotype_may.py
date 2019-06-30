from pysnptools.snpreader import Bed
import numpy as np
import hickle as hkl
import pandas as pd
import argparse
import tqdm


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Remove not existing patients in calls')

    parser.add_argument('--snps', dest='snps',
                        help='bed fam files prefix',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/dontPush/calls/ukb_cal_chr1_v2', type=str)

    parser.add_argument('--phenos', dest='phenos',
                        help='data to be cleaned',
                        default=None,
                        type=str)

    parser.add_argument('--to_dir', dest='to_dir',
                        help='path and name where to seve cleaned csv',
                        default='/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/cleaned.csv',
                        type=str)

    args = parser.parse_args()
    return args


# 100 patients with all snps each
dataset = np.empty(shape=(25, 63480))  # 63487


def data_on(ondisk, patients_):
    for row in tqdm.tqdm(range(25)):
        pat = patients_[row]
        pat_id = ondisk.iid_to_index(
            [[str(pat).encode(), str(pat).encode()]])  # get the id to get the value, this is an array
        offset = 0
        values = ondisk[pat_id[0], :63480].read(order='A', view_ok=True).val  # save in ram one by one
        for snp in values[0]:  # snp_val it's a 2 dimentional np array so 0 is needed
            if not np.isnan(snp):  # zero by default, but insert 1 where the geno is
                assert -1 < snp < 3  # only 0, 1, and 2 or minor homozygous, heterozygous and major homozygous
                dataset[row][offset] = snp
            else:
                dataset[row][offset] = 5  # substitute nan's by 5
            offset += 1


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    snps = Bed(args.snps, count_A1=False)  # count_A1 counts the allels numbers
    phenos = pd.read_csv('/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/cleaned.csv', sep=',')[-25:]
    phenos = phenos.reset_index()

    iid_patients = phenos.loc[:, 'f.eid']

    data_on(ondisk=snps, patients_=iid_patients)

    print("making the geno file...")
    # pd.DataFrame(dataset).to_csv("/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/genos.csv", sep=' ', header=None, index=False)
    hkl.dump(dataset, "/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_val_25.hkl", mode='w')
