

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Datasets
train_dataset = datasets.MNIST(root="data/", train=True, download=True, transform=train_transform)
test_dataset  = datasets.MNIST(root="data/", train=False, download=True, transform=test_transform)

# Define the CNN model
class MNISTClassifier(nn.Module):
    """
    Simple CNN for MNIST:
      - Conv(1→32) + ReLU
      - Conv(32→64) + ReLU + MaxPool
      - FC(64*14*14 → 128) + ReLU
      - FC(128 → 10)
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2)
        self.fc1   = nn.Linear(64 * 14 * 14, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Training and evaluation functions
def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(dataloader, model, loss_fn):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return test_loss / len(dataloader), correct / total

# Optuna objective
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Prepare DataLoaders for this trial
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = MNISTClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train for a few epochs
    for _ in range(5):
        train_epoch(train_loader, model, loss_fn, optimizer)

    # Validate
    _, accuracy = evaluate(test_loader, model, loss_fn)
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print(f"Best accuracy: {study.best_value*100:.2f}%")