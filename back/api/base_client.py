import flwr as fl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from model import SimpleCNN
import time
import uuid
import random

TOTAL_DATASET_SIZE = 1000  # Total number of images to use from MNIST
DATASET_PERCENTAGE = 100
MODEL_ID = ''
ETH_ADDRESS = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Select a random subset of TOTAL_DATASET_SIZE from the full dataset
indices = torch.randperm(len(full_dataset))[:TOTAL_DATASET_SIZE]
subset_dataset = Subset(full_dataset, indices)

# Calculate the number of samples to use based on the percentage
num_samples = int(len(subset_dataset) * DATASET_PERCENTAGE / 100)

# Create a stratified subset
class_indices = [[] for _ in range(10)]
for idx in indices:
    _, label = full_dataset[idx]
    class_indices[label].append(idx)

subset_indices = []
for class_idx in class_indices:
    # Ensure that the sampling is within the bounds of each class's available indices
    num_samples_per_class = int(len(class_idx) * DATASET_PERCENTAGE / 100)
    subset_indices.extend(random.sample(class_idx, num_samples_per_class))

# Correctly create the dataset from the subset indices
dataset = Subset(full_dataset, subset_indices)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

class MNISTClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.client_id = str(uuid.uuid4())

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(device) for k, v in params_dict}
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        start_time = time.time()
        for epoch in range(1):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        compute_time = time.time() - start_time
        return self.get_parameters(), len(train_loader.dataset), {"compute_time": compute_time, "model_id": MODEL_ID, "eth_address": ETH_ADDRESS}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(val_loader.dataset)
        return loss, len(val_loader.dataset), {"accuracy": accuracy, "model_id": MODEL_ID, "eth_address": ETH_ADDRESS}

fl.client.start_client(server_address="localhost:8080", client=MNISTClient().to_client())