import torch, os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def get_data(data_path="../data/SeoulBikeData.csv",testData = False):
    assert os.path.isfile(data_path), f"{os.path.realpath(data_path)} : File not exist"

    df = pd.read_csv(data_path)
    df = df.select_dtypes(include=[np.number]) #select columns with numerical data
    print(df.shape)

    if testData :
        xtrain, xtest, ytrain, ytest = train_test_split(df.drop(["Rented Bike Count"],axis=1).values, df["Rented Bike Count"].values,test_size=0.20, random_state=42)
        return DataLoader(TensorDataset(torch.from_numpy(xtrain),torch.from_numpy(ytrain)), batch_size=30), DataLoader(TensorDataset(torch.from_numpy(xtest),torch.from_numpy(ytest)))

    y = torch.from_numpy(df["Rented Bike Count"].values).float()
    x = torch.from_numpy(df.drop(["Rented Bike Count"],axis=1).values).float()

    return DataLoader(TensorDataset(x,y), batch_size=30)

_,_ = get_data(testData=True)

# if __name__ == '__main__':
#     get_train_data()
