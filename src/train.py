from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from context import scripts, BaseModel, DLModel
import scripts
from scripts.getData import get_data_collab, dl_preprocess_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # get device for training model


def main(args):
    torch.manual_seed(42)

    # Load data
    train, test = get_data_collab(Path("../data/collaborative-filtering/"))

    # # Train base model
    base_model = BaseModel()
    print("Base model training started")
    base_model.train(train)
    print("Base model training ended")
    base_model.save(Path.cwd())

    # Neural model
    train_batches, users_num, movies_num = dl_preprocess_data(train, batch_size=256)
    nn_model = DLModel(users_num, movies_num)
    print(nn_model)

    # NN training
    epochs_num = 10
    criterion = torch.nn.MSELoss(reduction="mean")
    optim = torch.optim.Adam(nn_model.parameters(), lr=1e-3)

    for epoch in range(epochs_num):
        epoch_loss = 0
        for users, movies, ratings in train_batches:
            users.to(device)
            movies.to(device)
            ratings.to(device)

            nn_model.zero_grad()
            output = nn_model.forward(users, movies).squeeze()
            loss = criterion(ratings, output)
            loss.backward()
            optim.step()
            epoch_loss += loss
        error = epoch_loss / len(train_batches)
        print(f"Epoch {epoch + 1}: loss: {error:.5f}")

    nn_model.save(Path().cwd())


if __name__ == '__main__':
    main(None)
