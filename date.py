import pandas as pd
import torch

# citim tabelul: fiecare rand e o ora, fiecare coloana o statie
df = pd.read_csv("Ref-Data/NO2_O3_all.csv", sep=";")

# print(df.head())  # primele 5 randuri
# print(df.shape)  # (cate randuri, cate coloane)
# print(df.columns)  # numele celor 22 de coloane
# print(df.isna().sum())  # cate valori lipsesc in fiecare coloana

# keep only the complete hours (all 21 stations have a value)
df = df.dropna()  # remove every row with at least one missing value
print(df.shape)  # how many hours are left

# remove repeated hours (the same timestamp appearing twice)
df = df.drop_duplicates(subset=["timestamp"])

# turn the text "2015-01-01T01:00" into a real date and time
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M")

# use the timestamp as the row label, so only the 21 stations remain as columns
df = df.set_index("timestamp")

print(df.shape)

# training data: only the year 2023
df_train = df["2023-01-01":"2023-12-31"]
print(df_train.shape)

# standardization: x_j = (x_j_raw - m_j) / s_j
m = df_train.mean()  # m_j: mean of each station (21 numbers)
s = df_train.std()  # s_j: standard deviation of each station (21 numbers)
X_train = (df_train - m) / s


# from a pandas table to a PyTorch tensor (a 6270 x 21 matrix)
X_train = torch.tensor(X_train.values, dtype=torch.float32)

# check
print(X_train.shape)  # should be torch.Size([6270, 21])
print(X_train[0])  # the first hour: one real x
