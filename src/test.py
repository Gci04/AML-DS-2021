from context import scripts
import scripts
from torch import nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score

class NeuralNetworkClassification(nn.Module):
    def __init__(self,input_dim=10,output_dim=1):
        super(NeuralNetworkClassification, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 25)
        self.output = nn.Linear(25, output_dim)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for inputs, targets in test_dl:
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


if __name__ == '__main__':
    _,testDataLoader = scripts.get_data(data_path="../data/SeoulBikeData.csv",testData = True)
    model = NeuralNetworkClassification(input_dim=9)
    model.load_state_dict(torch.load("./trained_model.pth"))

    model.eval()
    model_acc = evaluate_model(testDataLoader,model)
    print("Model Accuracy : {:.1f}%".format(model_acc*100))
