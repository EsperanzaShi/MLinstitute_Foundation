import ssl, certifi
# Ensure HTTPS downloads use certifi's CA bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters (from Optuna tuning)
lr = 0.04156575742501107  # optimized from CNN tuning
batch_size = 32
epochs = 20

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformation and loading with augmentation
# Augmented training transform
train_transform = transforms.Compose([
    transforms.RandomRotation(15),                       # random rotation ±15°
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), shear=10),  # random shift and shear
    transforms.RandomPerspective(distortion_scale=0.2),  # slight perspective distortions
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1]
])
# Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="data/",
                               train=True,
                               download=True,
                               transform=train_transform)
test_dataset  = datasets.MNIST(root="data/",
                               train=False,
                               download=True,
                               transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# Define the neural network model (CNN)
class MNISTClassifier(nn.Module):
    """
    Simple CNN for MNIST:
      - Conv(1→32) + ReLU
      - Conv(32→64) + ReLU + MaxPool
      - FC(64*14*14 → 128) + ReLU
      - FC(128 → 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 14 * 14, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))            # → [batch, 32, 28, 28]
        x = F.relu(self.conv2(x))            # → [batch, 64, 28, 28]
        x = self.pool(x)                     # → [batch, 64, 14, 14]
        x = x.view(x.size(0), -1)            # → [batch, 64*14*14]
        x = F.relu(self.fc1(x))              # → [batch, 128]
        return self.fc2(x)                   # → [batch, 10]

# Training and evaluation functions
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


def evaluate(dataloader, model, loss_fn):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return test_loss / len(dataloader), correct / total

# Main training loop
if __name__ == "__main__":
    # Initialize model, loss, and optimizer
    model     = MNISTClassifier().to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        test_loss, test_acc = evaluate(test_loader, model, loss_fn)
        print(f"Epoch {epoch}/{epochs} — "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc*100:>5.2f}%")
        # Track best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "mnist_best.pth")

    print(f"Training complete. Best Test Accuracy: {best_acc*100:.2f}%")
    print("Saved best model to mnist_best.pth")
