import polars as pl

import torch
import torch.nn as nn
import numpy as np

TARGET_LAG = 1 # target lag

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

def calculate_lag_correlations(df, lags):
    """
    Calculates and returns the lagged correlations between different tokens' prices in the dataset.

    Parameters:
    - df: The input dataframe containing at least 'token', 'date', and 'price' columns.
    - lags (optional): A list of integers specifying the lag days for which to calculate correlations.

    The function iterates over each unique token pair in the dataset, computes the correlation of their prices at
    specified lags, and returns a dictionary with these correlations. The dictionary keys are formatted as
    "{base_token}_vs_{compare_token}" with sub-keys for each lag indicating the correlation at that lag.

    Returns:
    A dictionary of lagged price correlations for each token pair in the dataset.
    """

    correlations = {}
    tokens = df.select("token").unique().to_numpy().flatten()
    for base_token in tokens:
        for compare_token in tokens:
            if base_token == compare_token:
                continue
            base_df = df.filter(pl.col("token") == base_token).select(["date", "price"]).sort("date")
            compare_df = df.filter(pl.col("token") == compare_token).select(["date", "price"]).sort("date")
            merged_df = base_df.join(compare_df, on="date", suffix="_compare")
            key = f"{base_token}_vs_{compare_token}"
            correlations[key] = {}
            for lag in lags:
                merged_df_lagged = merged_df.with_columns(pl.col("price_compare").shift(lag))
                corr_df = merged_df_lagged.select(
                    pl.corr("price", "price_compare").alias("correlation")
                )
                corr = corr_df.get_column("correlation")[0]
                correlations[key][f"lag_{lag}_days"] = corr
                
    return correlations

def main_dataset_manipulation():
    """
    """

def apy_dataset_manipulation():
    """
    """

def tvl_dataset_manipulation():
    """
    """

def load_and_df_preprocessing():
    """
    Loads and processes the main, APY, and TVL datasets, joining them on the date column and performing postprocessing.

    Returns:
    A DataFrame ready for further analysis or model training, containing combined and processed features from all datasets.
    """

    df_main = main_dataset_manipulation()
    apy_df = apy_dataset_manipulation()
    tvl_df = tvl_dataset_manipulation()

    df_main = df_main.join(tvl_df, on = "date", how = "inner")
    df_main = df_main.join(apy_df, on = "date", how = "inner")

    num_rows_to_select = len(df_main) - TARGET_LAG
    df_main = df_main.slice(0, num_rows_to_select)

    # Some of the extra tokens we added do not have much historical information, so we raised the minimum date of our dataset a little bit.
    df_main = df_main.filter(pl.col("year") >= 2022)
    df_main = df_main.drop(["token","market_cap"])
    df_main = delete_null_columns(df_main, 0.2)
    return df_main
