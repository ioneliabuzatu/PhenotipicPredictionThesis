import os
import numpy as np
import hickle as hkl
import argparse  # maybe will use it later
# from config import

from transformingdata import get_original

try:
    # os._exists('dontPush/testDELETE.hkl')
    file_hkl = hkl.load('dontPush/testCNN850X6k.hkl')
    print(file_hkl.shape)
except OSError:
    getX012 = get_original()
    hkl.dump(getX012, 'dontPush/testCNN850X6k.hkl', mode='w')
