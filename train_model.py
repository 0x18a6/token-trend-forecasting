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

def predict(model, X_test):
    """
    Makes predictions using a trained model on the provided test data.

    Parameters:
    - model: The trained neural network model.
    - X_test: Test data features as a NumPy array.

    Returns:
    Predictions as a NumPy array.
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    return y_pred_tensor.numpy()

def prepare_train_test(df_train, df_test):
    """
    Preprocesses training and test dataframes by standardizing columns based on training data statistics.

    Parameters:
    - df_train: The training dataframe.
    - df_test: The test dataframe.

    Returns:
    A tuple containing the preprocessed training and test dataframes.
    """
    for col in df_train.columns:
        mean_val = df_train[col].mean()
        std_dev = df_train[col].std() if df_train[col].std() != 0 else 1

        df_train = df_train.with_columns(((df_train[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
        df_test = df_test.with_columns(((df_test[col].fill_null(mean_val) - mean_val) / std_dev).alias(col))
    return df_train, df_test

def delete_null_columns(df, null_percentage):
    """    
    Removes columns from a dataframe where the percentage of null values exceeds a specified threshold.

    Parameters:
    - df: The dataframe to process.
    - null_percentage: The threshold percentage of null values for column removal.

    Returns:
    The dataframe with columns removed based on the null value threshold.
    """
    threshold = df.shape[0] * null_percentage
    columns_to_keep = [
        col_name for col_name in df.columns if df[col_name].null_count() <= threshold
    ]
    return df.select(columns_to_keep)