import pandas as pd

file = pd.read_csv("dontPush/pheno10000.csv", sep="\t")
file = file.astype(int)
file = file.astype('category')
print(file.iloc[:, 1])

