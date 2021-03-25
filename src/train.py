import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from context import scripts
import scripts

device = 'cuda' if torch.cuda.is_available() else 'cpu' # get device for training model
writer = SummaryWriter()

class NeuralNetworkRegression(nn.Module):
    def __init__(self,input_dim=10,output_dim=1):
        super(NeuralNetworkRegression, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 25)
        self.output = nn.Linear(25, output_dim)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.output(x)
        return x

def main(args):

    #load data
    train_DataLoader = scripts.get_data(data_path="../data/SeoulBikeData.csv",testData = False)
    # train_DataLoader = DataLoader(TensorDataset(train_x, train_y), batch_size=30)

    model = NeuralNetworkRegression(input_dim=9).float()

    lr = 1e-1
    n_epochs = 100
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    size = len(train_DataLoader.dataset)
    for epoch in range(n_epochs):
        l = 0.0
        for X,y in train_DataLoader:
            model.train() #set model to training mode
            ypred = model(X) #Foward pass
            loss = loss_fn(ypred,y.float()) # Calcutation of loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            l += (loss/size)

        writer.add_scalar("train_loss", l, epoch)
        for tag, parm in model.named_parameters():
            writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

if __name__ == '__main__':
    main(None)
