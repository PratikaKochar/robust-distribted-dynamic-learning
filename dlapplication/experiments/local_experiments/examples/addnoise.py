import pandas as pd
df2 = pd.read_csv('cifar_trainfinal.csv')
print(df2.head(5))
#clean_signal = pd.DataFrame([[1,2],[3,4]], columns=list('AB'), dtype=float)
import numpy as np
mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, df2.shape)
signal = df2 + noise
print(signal.head(5))
signal.to_csv("noisycifar100.csv", index=False)