# ~13k not in the calls from tab

import pandas as pd
from pysnptools.snpreader import Bed
import argparse


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


def cleaner():

    snps = Bed(args.snps, count_A1=False)
    patients = pd.read_csv('/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_clean.csv', sep=' ',
                           index_col=0)

    patients_id = pd.read_csv('/Users/ioneliabuzatu/PycharmProjects/biobank/obesity/data/bmi_clean.csv', sep=' ')
    pats = patients_id.iloc[:, 0]
    count_not_there = 0

    for p in pats:
        search = str(p).encode('ascii')
        try:
            hi = snps.iid_to_index([[search, search]])
        except:
            patients = patients.drop([p])
            count_not_there += 1
            print(patients.shape)

    return patients


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    # getting the clened csv and save it to csv
    print('Saving the cleaned csv file...')
    patients = cleaner()
    patients.to_csv(args.to_dir)
