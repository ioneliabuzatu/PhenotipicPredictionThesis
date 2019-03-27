import os
from pysnptools.snpreader import Bed, Pheno
import numpy as np
import hickle as hkl
import os
import h5py
import argparse  # maybe will use it later
# from config import
import tqdm
from transformingData import getX01, getXoriginal

try:
    # os._exists('dontPush/testDELETE.hkl')
    file_hkl = hkl.load('dontPush/testDELETE.hkl')
    print(file_hkl.shape)
except OSError:
    getX01 = getX01()
    hkl.dump(getX01, 'dontPush/testDELETE.hkl', mode='w')

original = getXoriginal()
# hkl.dump(original, 'dontPush/test_geno100X500_012.hkl', mode='w')
