### INIT
# packages
import pandas as pd
import numpy as np

# read data
df = pd.read_csv('../LAOMLProject1/EastWestAirlinesCluster.csv')
df = df.drop(columns=['ID#'])
normalized_df = (df-df.min()) / (df.max()-df.min())

# define parameters
K = 4
ncol = 11

### MAIN
# initialise clusters
C = np.random.uniform(low=0, high=1, size=(ncol,K))

#


# functions

def kernel_euclidean(x,y)
    return np.linalg.norm(x-y)

def kernel_poly(x,y,r)
    return (1 + np.dot(x,y))^r

def kernel_gauss(x,y,gamma)
    return np.exp(-gamma*np.linalg.norm(x-y)^2)

# test
print(normalized_df)
print(normalized_df.columns)