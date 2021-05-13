from pathlib import Path
import time
import torch

from context import scripts, BaseModel, DLModel
import scripts
from scripts import get_data_collab, dl_preprocess_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # get device for training model

if __name__ == '__main__':
    train, test = get_data_collab(Path("../data/collaborative-filtering/"))

    # Base model testing
    base_model = BaseModel()
    base_model.load(Path.cwd())
    print(f"BaseModel test MSE: {base_model.evaluate(test)}")

    # NN model testing
    _, users_num, movies_num = dl_preprocess_data(train, batch_size=128)
    test, _, _ = dl_preprocess_data(test, batch_size=128)
    nn_model = DLModel(users_num, movies_num)
    nn_model.load(Path.cwd(), device)
    print(f"DLModel test MSE: {nn_model.evaluate(test, device)}")

    start_time = time.time()
    print("Top-5 movie suggestions of base model for user 1: ", base_model.predict(user_id=1))
    print(f"BaseModel inference time: {time.time() - start_time}")
    start_time = time.time()
    print("Top-5 movie suggestions of NN model for user 1: ", nn_model.predict(user_id=1, movies_num=movies_num))
    print(f"NN model inference time: {time.time() - start_time}")
