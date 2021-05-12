from pathlib import Path

import pandas as pd

import torch

# Get training and testing set
def get_data_collab(path: Path):
    train = pd.read_csv(path / "train.csv")
    test = pd.read_csv(path / "test.csv")
    return train, test


def dl_preprocess_data(data, batch_size=64):
    user_ids = data['userId'].values - 1
    movie_ids = data['movieId'].values - 1
    ratings = data['rating'].values
    users_num, movies_num = max(user_ids) + 1, max(movie_ids) + 1
    batches = []
    for i in range(0, len(ratings), batch_size):
        offset = min(batch_size + i, len(ratings))
        batches.append((
            torch.tensor(user_ids[i: offset], dtype=torch.long),
            torch.tensor(movie_ids[i: offset], dtype=torch.long),
            torch.tensor(ratings[i: offset], dtype=torch.float)
        ))

    return batches, users_num, movies_num


if __name__ == '__main__':
    pass
    # _,_ = get_data(testData=True)
