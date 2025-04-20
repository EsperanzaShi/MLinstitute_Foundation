import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import optuna
"""
trainmodel.py

This script trains a neural network model to classify handwritten digits from the MNIST dataset. 
It includes data loading, model definition, training, evaluation, and saving the trained model.

Modules:
- torch: PyTorch library for building and training neural networks.
- torchvision: Provides datasets and transformations for computer vision tasks.
- torch.utils.data: Utilities for data loading and batching.

Functions:
- train_epoch(dataloader, model, loss_fn, optimizer): Trains the model for one epoch.
- evaluate(dataloader, model, loss_fn): Evaluates the model on the test dataset.

Classes:
- MNISTClassifier: Defines a simple feedforward neural network for MNIST classification.

Usage:
- Run the script to train the model for 5 epochs and save the trained model as "mnist_model.pth".
"""
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Augmented training transform
train_transform = transforms.Compose([
    transforms.RandomRotation(15),                             # random rotation ±15°
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),  # random shift and shear
    transforms.RandomPerspective(distortion_scale=0.2),        # slight perspective distortions
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))                     # scale to [-1,1]
])
# Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download & load training and test sets
train_dataset = datasets.MNIST(root="data/",
                               train=True,
                               download=True,
                               transform=train_transform)

test_dataset  = datasets.MNIST(root="data/",
                               train=False,
                               download=True,
                               transform=test_transform)

# Define the neural network model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
# Defines how input x flows through the network. 
# Returning raw logits allows you to pair it with CrossEntropyLoss, which applies softmax internally.
    def forward(self, x):       
        return self.fc(x)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model  = MNISTClassifier().to(device)
loss_fn   = nn.CrossEntropyLoss()

def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

#evaluation loop
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
            total   += y.size(0)
    return test_loss / len(dataloader), correct / total

# Optuna hyperparameter optimization
def objective(trial):
    # Suggest hyperparameters
    lr         = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Re‑create model, optimizer, and data‑loaders for each trial
    model     = MNISTClassifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Train for a small, fixed number of epochs
    for _ in range(5):
        train_epoch(train_loader, model, loss_fn, optimizer)

    # Evaluate and return the accuracy (Optuna will maximize this)
    _, accuracy = evaluate(test_loader, model, loss_fn)
    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print(f"Best accuracy: {study.best_value*100:.2f}%")