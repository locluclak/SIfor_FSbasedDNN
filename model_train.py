import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# global device
# device = torch.device("cuda")
# Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)


class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)


# Generate synthetic data
def generate_data(n, p, col_means=None, col_stds=None):
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42,
    )
    if col_means is not None and col_stds is not None:
        for i in range(p):
            X[:, i] = X[:, i] * col_stds[i] + col_means[i]
    return X, y


# Load and preprocess data
def load_and_preprocess_data(X, y, batch_size=32):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, X_test_tensor, y_test_tensor


# Training function
def train_model(
    model, train_loader, test_loader, epochs=200, lr=0.002, de=torch.device("cpu")
):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(de)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(de), y_batch.to(de)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(de), y_batch.to(de)
                    outputs = model(X_batch)
                    test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += y_batch.size(0)
                    test_correct += (predicted == y_batch).sum().item()
                test_loss /= len(test_loader.dataset)
                test_accuracy = test_correct / test_total
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}"
            )


def gendata_trainmodel(n_samples=2000, n_features=10, train=False, device = torch.device("cpu")):
    col_means = [0] * n_features
    col_stds = [1] * n_features
    X, y = generate_data(n_samples, n_features, col_means, col_stds)

    # Load and preprocess data
    train_loader, test_loader, X_test, y_test = load_and_preprocess_data(
        X, y, batch_size=32
    )

    # Initialize model
    input_dim = X_test.shape[1]
    model = Classifier(input_dim).to(device)

    # Train or load model
    save_model = "./weights/classify_model.pth"
    if train or not os.path.exists(save_model):
        print("Training the model...")
        train_model(
            model, train_loader, test_loader, epochs=200, lr=0.002, de=device
        )
        torch.save(model.state_dict(), save_model)
        print(f"Model saved as '{save_model}'")
    else:
        print("Loading the model...")
        model.load_state_dict(torch.load(save_model, map_location=device))
        model.eval()
        print("Model loaded successfully")
    return {"model": model, "test_data": (X_test, y_test)}
