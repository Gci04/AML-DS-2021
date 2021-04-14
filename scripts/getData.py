import torch, os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data_name(path):
    def padding(char_list):
        char_list.extend([0] * (embedding_num - len(char_list)))
        return char_list

    data = pd.read_csv(path)
    data.sort_values(by="Name", key=lambda col: col.apply(lambda x: len(x)), inplace=True)
    unique = list(set("".join(data["Name"])))
    unique.sort()
    vocab = dict(zip(unique, range(1, len(unique) + 1)))

    # Get maximum number of characters
    embedding_num = data["Name"].apply(lambda x: len(x)).max()

    data["Name"] = data["Name"].apply(lambda string: padding([vocab[x] for x in list(string)]))
    # One is M and Zero is F
    data["Gender"] = data["Gender"].apply(lambda x: 1 if x == "M" else 0)
    return data, vocab


if __name__ == '__main__':
    pass
    # _,_ = get_data(testData=True)
