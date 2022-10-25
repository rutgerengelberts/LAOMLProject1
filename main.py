# packages
import pandas as pd

# read data
df = pd.read_csv('../LAOMLProject1/EastWestAirlinesCluster.csv')
df = df.drop(columns=['ID#'])
normalized_df = (df-df.min()) / (df.max()-df.min())

# test

print(normalized_df)
print(normalized_df.columns)