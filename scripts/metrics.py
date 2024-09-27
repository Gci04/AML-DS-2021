import torch


def accuracy(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            name, labels = data
            outputs = model(name)
            predicted = torch.round(torch.sigmoid(outputs.data))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
