from pathlib import Path
from typing import Optional

from fastapi import FastAPI

import pandas as pd
from numpy import random, multiply, load, save
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error


# Get training and testing set
def get_data_collab(path: Path):
    train = pd.read_csv(path / "train.csv")
    test = pd.read_csv(path / "test.csv")
    return train, test


class BaseModel:
    def train(self, data, lr=1e-4, iterations=100, w=1e-4, k=5):
        """
        Train the model using matrix factorization
        """
        ratings = data['rating'].values
        user_ids = data['userId'].values
        movie_ids = data['movieId'].values

        sparse_matrix = coo_matrix((ratings, (user_ids, movie_ids)))
        self.P_matrix = random.rand(sparse_matrix.shape[0], k)
        self.Q_matrix = random.rand(sparse_matrix.shape[1], k)

        # Errors to return. Used for plotting
        errors = []
        for i in range(iterations):
            P_curr = self.P_matrix[user_ids, :]
            Q_curr = self.Q_matrix[movie_ids, :]

            preds = multiply(P_curr, Q_curr).sum(axis=1)
            diff = coo_matrix((preds, (user_ids, movie_ids))) - sparse_matrix

            # Update matrices
            self.P_matrix = self.P_matrix - lr * (w * self.P_matrix + diff @ self.Q_matrix)
            self.Q_matrix = self.Q_matrix - lr * (w * self.Q_matrix + diff.T @ self.P_matrix)
            err = mean_squared_error(ratings, preds)
            errors.append(err)
            print(f"Iter {i + 1}: loss: {err:.5f}")
        return errors

    def evaluate(self, data):
        """
        Evaluates model on given data.
        Returns MSE error
        """
        ratings = data['rating'].values
        user_ids = data['userId'].values
        movie_ids = data['movieId'].values

        P_curr = self.P_matrix[user_ids, :]
        Q_curr = self.Q_matrix[movie_ids, :]

        preds = multiply(P_curr, Q_curr).sum(axis=1)

        return mean_squared_error(ratings, preds)

    def predict(self, user_id, top_k=5):
        """
        Prediction byt the user_id
        """

        def clipping(val):
            # Filtering values lower than 1 and higher than 5
            if val > 5:
                return 5
            elif val < 1:
                return 1
            return val

        preds = [(i, clipping(x)) for i, x in enumerate(self.Q_matrix @ self.P_matrix[user_id].T)]

        return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]

    def load(self, path: Path):
        self.P_matrix = load(str((path / "P_matrix.npy")))
        self.Q_matrix = load(str((path / "Q_matrix.npy")))

    def save(self, path: Path):
        save(str((path / "P_matrix.npy")), self.P_matrix)
        save(str((path / "Q_matrix.npy")), self.Q_matrix)


# Load data
train, test = get_data_collab(Path("collaborative-filtering"))

# # Train base model
base_model = BaseModel()
print("Base model training started")
base_model.train(train)
print("Base model training ended")
base_model.save(Path.cwd())

app = FastAPI()


@app.get("/recommendation")
def read_root(user_id: int):
    try:
        preds = base_model.predict(user_id=user_id)
    except Exception as e:
        return {"error": e.args[0]}

    return dict(preds)


