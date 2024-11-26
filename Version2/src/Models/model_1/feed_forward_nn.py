import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetworkModel:
    def __init__(self, input_size, hidden_size=64, learning_rate=0.001):
        """
        Initialize the Neural Network Model.
        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param learning_rate: Learning rate for the optimizer.
        """
        # Define the architecture of the neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output layer
        )

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scaler for feature normalization
        self.scaler = StandardScaler()

    def train(self, data_frame, feature_columns, target_column, epochs=100, batch_size=16, test_size=0.2):
        """
        Train the neural network model.
        :param data_frame: DataFrame containing the data.
        :param feature_columns: List of feature column names.
        :param target_column: Name of the target column.
        :param epochs: Number of training epochs.
        :param batch_size: Size of each training batch.
        :param test_size: Fraction of data to use for testing.
        """
        # Split data into features and target
        X = data_frame[feature_columns].values
        y = data_frame[target_column].values

        # Normalize features
        X = self.scaler.fit_transform(X)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Training loop
        for epoch in range(epochs):
            # Shuffle the training data into batches
            permutation = torch.randperm(X_train.size(0))
            epoch_loss = 0.0

            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Print epoch progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """
        Evaluate the neural network model on the test dataset.
        :return: Dictionary with evaluation metrics (MSE, RMSE, MAE, R2).
        """
        # Predict on the test set
        with torch.no_grad():
            predictions = self.model(self.X_test).numpy().flatten()
            y_test = self.y_test.numpy().flatten()

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def cross_validate(self, data_frame, feature_columns, target_column, k=5, epochs=100):
        """
        Perform k-fold cross-validation on the neural network model.
        :param data_frame: DataFrame containing the data.
        :param feature_columns: List of feature column names.
        :param target_column: Name of the target column.
        :param k: Number of folds for cross-validation.
        :param epochs: Number of training epochs for each fold.
        :return: List of evaluation metrics for each fold.
        """
        X = data_frame[feature_columns].values
        y = data_frame[target_column].values

        # Normalize features
        X = self.scaler.fit_transform(X)

        # Perform k-fold cross-validation
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Convert data to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

            # Reinitialize the model for each fold
            self.__init__(input_size=X_train.shape[1])

            # Train the model
            for epoch in range(epochs):
                permutation = torch.randperm(X_train.size(0))
                for i in range(0, X_train.size(0), 16):  # Mini-batch size = 16
                    indices = permutation[i:i + 16]
                    batch_X, batch_y = X_train[indices], y_train[indices]

                    predictions = self.model(batch_X)
                    loss = self.criterion(predictions, batch_y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Evaluate the model
            with torch.no_grad():
                predictions = self.model(X_test).numpy().flatten()
                y_test = y_test.numpy().flatten()

                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

            fold_metrics.append({"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2})
            print(f"Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        return fold_metrics

    def visualize_errors(self):
        """
        Visualize error patterns using a residual plot.
        """
        with torch.no_grad():
            predictions = self.model(self.X_test).numpy().flatten()
            residuals = self.y_test.numpy().flatten() - predictions

            # Plot residuals
            plt.figure(figsize=(8, 5))
            plt.scatter(predictions, residuals, alpha=0.7)
            plt.axhline(0, color='red', linestyle='--', linewidth=2)  # Zero error line
            plt.xlabel("Predicted Suitability Score")
            plt.ylabel("Residual (True - Predicted)")
            plt.title("Residual Plot")
            plt.show()