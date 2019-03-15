import argparse
import pandas as pd
from config import *

data = pd.read_csv('regression/full_phenotype_may', sep=" ")
test = data.loc[:end, ["f.4079.0.0"]]
test = test.fillna(test.mean())
test.to_csv(filename, sep='\t')

