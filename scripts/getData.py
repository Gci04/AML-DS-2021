import torch, os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(data_path="../data/SeoulBikeData.csv",testData = False):
    assert os.path.isfile(data_path), f"{os.path.realpath(data_path)} : File not exist"

    df = pd.read_csv(data_path, engine='python')
    df = df.select_dtypes(include=[np.number]) #select columns with numerical data
    df["target"] = df["Rented Bike Count"].apply(lambda x : 1 if  x > 500 else 0)
    df.drop(["Rented Bike Count"],axis=1,inplace=True)

    if testData :
        xtrain, ytrain = df.drop(["target"],axis=1).values[:7008], df["target"].values[:7008].reshape(-1,1)
        xtest, ytest = df.drop(["target"],axis=1).values[7008:], df["target"].values[7008:].reshape(-1,1)
        # xtrain, xtest, ytrain, ytest = train_test_split(df.drop(["target"],axis=1).values[7008], df["target"].values.reshape(-1,1),test_size=0.20, random_state=42)
        scaler = StandardScaler().fit(xtrain)
        xtrain = torch.from_numpy(scaler.transform(xtrain)).float()
        xtest = torch.from_numpy(scaler.transform(xtest)).float()
        return DataLoader(TensorDataset(xtrain,torch.from_numpy(ytrain).float()), batch_size=30), DataLoader(TensorDataset(xtest,torch.from_numpy(ytest).float()))

    scaler = StandardScaler().fit(df.drop(["target"],axis=1).values)
    x = scaler.transform(df.drop(["target"],axis=1).values)
    y = torch.from_numpy(df["target"].values.reshape(-1,1)).float()
    x = torch.from_numpy(x).float()

    return DataLoader(TensorDataset(x,y), batch_size=30)


if __name__ == '__main__':
    pass
    # _,_ = get_data(testData=True)
