from torch import nn
import torch
import torch.utils.data as data_utils

from context import scripts
import scripts


class NeuralNetworkClassification(nn.Module):
    def __init__(self, vocab_len, input_dim=15, output_dim=1):
        super(NeuralNetworkClassification, self).__init__()
        self.embedding = nn.Embedding(vocab_len + 1, input_dim)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_dim ** 2, 25)
        self.output = nn.Linear(25, output_dim)

    def forward(self, x):
        x = self.flatten(self.embedding(x))
        x = torch.relu(self.layer1(x))
        x = self.output(x)
        return x


class BaselineLSTM(nn.Module):
    def __init__(self, vocab_len, embedding_dim=5, input_dim=15, output_dim=1):
        super(BaselineLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_len + 1, embedding_dim=embedding_dim)
        self.lstm_layer = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=5,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim * embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm_layer(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class BetterLSTM(nn.Module):
    def __init__(self, vocab_len, embedding_dim=5, input_dim=15, output_dim=1):
        super(BetterLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_len + 1, embedding_dim=embedding_dim)
        self.lstm_layer = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=5,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=False)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim * embedding_dim, 20)
        self.linear2 = nn.Linear(20, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm_layer(x)
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    _, vocab = scripts.get_data_name("../data/data/train_eng.csv")
    test_data, _ = scripts.get_data_name("../data/data/test_eng.csv")

    target_test = torch.tensor(test_data["Gender"].values).to(torch.long)
    features_test = torch.Tensor(list(test_data["Name"].values)).to(torch.long)
    test = data_utils.TensorDataset(features_test, target_test)
    test_loader = data_utils.DataLoader(test)

    model_classical = NeuralNetworkClassification(len(vocab)).float()
    model_baseline = BaselineLSTM(len(vocab)).float()
    model_better_baseline = BetterLSTM(len(vocab)).float()

    model_classical.load_state_dict(torch.load("./model_classical.pth"))
    model_baseline.load_state_dict(torch.load("./model_baseline.pth"))
    model_better_baseline.load_state_dict(torch.load("./model_better_baseline.pth"))
    model_classical.eval()
    model_baseline.eval()
    model_better_baseline.eval()

    model_classical_acc = scripts.accuracy(model_classical, test_loader)
    model_baseline_acc = scripts.accuracy(model_baseline, test_loader)
    model_better_baseline_acc = scripts.accuracy(model_better_baseline, test_loader)
    print(f"Model Baseline Accuracy: {round(model_baseline_acc, 2)}%")
    print(f"Model Better LSTM Accuracy: {round(model_better_baseline_acc, 2)}%")
    print(f"Model Classical Accuracy: {round(model_classical_acc, 2)}%")
