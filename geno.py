import os
import numpy as np
import hickle as hkl
import argparse  # maybe will use it later
# from config import

from transformingData import getXoriginal

try:
    # os._exists('dontPush/testDELETE.hkl')
    file_hkl = hkl.load('dontPush/test10Kx10k.hkl')
    print(file_hkl.shape)
except OSError:
    getX012 = getXoriginal()
    hkl.dump(getX012, 'dontPush/test10Kx10k.hkl', mode='w')
