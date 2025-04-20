import os, sys
# Add this script's directory to sys.path so sibling modules can be imported
sys.path.append(os.path.dirname(__file__))
import torch
from mnist_classifier.trainmodel_CNN import MNISTClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuration
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Locate the project root relative to this script's directory
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path = os.path.join(top_dir, "mnist_best.pth")
output_dir = os.path.join(top_dir, "mnist_classifier")
output_file = os.path.join(output_dir, "confusion_matrix.png")

# 1. Load the trained model
model = MNISTClassifier().to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# 2. Prepare the MNIST test loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_ds = datasets.MNIST(root=os.path.join(top_dir, "data"), train=False,
                         download=False, transform=transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# 3. Collect true labels and predictions
y_true, y_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        logits = model(X)
        preds  = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())

# 4. Compute confusion matrix
labels = list(range(10))
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 5. Plot and save
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
ax.set_title("MNIST Confusion Matrix")
plt.tight_layout()

os.makedirs(output_dir, exist_ok=True)
disp.figure_.savefig(output_file)
print(f"Saved confusion matrix to {output_file}")