import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    """Simple neural network to forecast token trends"""
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc1())
        x = torch.relu(self.fc2())
        x = torch.sigmoid(self.fc3())

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    """
    Trains a neural network model using the specified training data, loss criterion, and optimizer.

    Parameters:
    - model: The neural network model to be trained.
    - criterion: The loss function used to evaluate the model's performance.
    - optimizer: The optimization algorithm used to update the model's weights.
    - X_train: Training data features as a NumPy array.
    - y_train: Training data labels/targets as a NumPy array.
    - epochs (optional): The number of training epochs (default is 100).

    Returns:
    The trained model.
    """
    model.train()
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.float32).reshape(-1, 1))

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
    return model
