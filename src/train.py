import torch
from sklearn.metrics import f1_score
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils

from context import scripts
import scripts

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # get device for training model
writer = SummaryWriter()


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


def train_model(model, train_DataLoader, test_DataLoader, device, writer, n_epochs=100, lr=1e-3):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    size = len(train_DataLoader)
    for epoch in range(n_epochs):
        l = 0.0
        f1 = 0.0
        for X, y in train_DataLoader:
            X.to(device)
            y.to(device)

            model.train()  # set model to training mode
            ypred = model(X)  # Forward pass
            loss = loss_fn(ypred, y.float().unsqueeze(1))  # Calcutation of loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            l += (loss / size)
            f1 += f1_score(y.float(), torch.round(torch.sigmoid(ypred.data)).squeeze(1), average="macro") / size

        curr_accuracy = scripts.accuracy(model, test_DataLoader)
        print(
            f"Epoch : {epoch + 1}, loss : {round(l.item(), 5)}, test accuracy: {round(curr_accuracy, 2)}, f1 average: {round(f1, 2)}")
        writer.add_scalar("train_loss", l, epoch)
        writer.add_scalar("test_accuracy", curr_accuracy, epoch)
        writer.add_scalar("f1-score", f1, epoch)
        for tag, parm in model.named_parameters():
            writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)


def main(args):
    # load data
    test_data, _ = scripts.get_data_name("../data/data/test_eng.csv")
    train_data, vocab = scripts.get_data_name("../data/data/train_eng.csv")

    batch_size = 1000

    targets = torch.tensor(train_data["Gender"].values).to(torch.long)
    features = torch.Tensor(list(train_data["Name"].values)).to(torch.long)

    target_test = torch.tensor(test_data["Gender"].values).to(torch.long)
    features_test = torch.Tensor(list(test_data["Name"].values)).to(torch.long)

    train = data_utils.TensorDataset(features, targets)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size)

    test = data_utils.TensorDataset(features_test, target_test)
    test_loader = data_utils.DataLoader(test)

    model_classical = NeuralNetworkClassification(len(vocab)).float()
    model_baseline = BaselineLSTM(len(vocab)).float()
    model_better_baseline = BetterLSTM(len(vocab)).float()

    train_model(model_baseline, train_loader, test_loader, device, SummaryWriter("runs/baseline"), n_epochs=30)
    train_model(model_classical, train_loader, test_loader, device, SummaryWriter("runs/classical"), n_epochs=30)
    train_model(model_better_baseline, train_loader, test_loader, device, SummaryWriter("runs/better"), n_epochs=30)

    torch.save(model_classical.state_dict(), "./model_classical.pth")
    torch.save(model_baseline.state_dict(), "./model_baseline.pth")
    torch.save(model_better_baseline.state_dict(), "./model_better_baseline.pth")


if __name__ == '__main__':
    main(None)
