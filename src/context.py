import sys
import os
from pathlib import Path

import torch
from numpy import random, multiply, load, save
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts


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


class DLModel(torch.nn.Module):

    def __init__(self, users_num, movies_num, embedding_dim=20, dropout=0.2):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(users_num, embedding_dim)
        self.movie_embedding = torch.nn.Embedding(movies_num, embedding_dim)
        self.l1 = torch.nn.Linear(2 * embedding_dim, 128)
        self.d1 = torch.nn.Dropout(dropout)
        self.l2 = torch.nn.Linear(128, 64)
        self.d2 = torch.nn.Dropout(dropout / 2)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, user_data, movie_data):
        features = torch.cat([self.user_embedding(user_data), self.movie_embedding(movie_data)], dim=1)
        x = torch.nn.ReLU()(self.d1(self.l1(features)))
        x = torch.nn.ReLU()(self.d2(self.l2(x)))
        output = torch.sigmoid(self.fc(x))

        # Min max scaling
        output = output * 4 + 1

        return output

    def evaluate(self, test, device):
        self.eval()
        criterion = torch.nn.MSELoss(reduction="mean")
        epoch_loss = 0
        for users, movies, ratings in test:
            users.to(device)
            movies.to(device)
            ratings.to(device)

            self.zero_grad()
            output = self.forward(users, movies).squeeze()
            loss = criterion(ratings, output)
            loss.backward()
            epoch_loss += loss
        return epoch_loss / len(test)

    def load(self, path: Path, device):
        self.load_state_dict(torch.load(path / "nn.pt", map_location=device))

    def save(self, path: Path):
        torch.save(self.state_dict(), path / "nn.pt")

    def predict(self, user_id, movies_num, top_k=5):
        preds = [(x, self.forward(torch.tensor(data=[user_id]), torch.tensor(data=[x])).squeeze().item()) for x in
                 range(movies_num)]

        return sorted(preds, key=lambda x: x[1], reverse=True)[:top_k]
