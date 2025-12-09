import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

brainFrame = pd.read_csv('brainsize.txt', delimiter='\t')
menDf = brainFrame[brainFrame['Gender'] == 'Male']
womenDf = brainFrame[brainFrame['Gender'] == 'Female']

menMeanSmarts = menDf[["PIQ", "FSIQ", "VIQ"]].mean(axis=1)
plt.scatter(menMeanSmarts, menDf["MRI_Count"])
mcorr = menDf[["PIQ", "FSIQ", "VIQ", "MRI_Count"]].corr()

sns.heatmap(mcorr)

plt.savefig('men_attribute_correlations.png')
