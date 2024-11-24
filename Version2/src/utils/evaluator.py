import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from scipy.stats import spearmanr
import torch

class Evaluator:
    def __init__(self, framework="sklearn"):
        """
        Initialize the Evaluator.
        :param framework: Framework used for the model ('sklearn', 'tensorflow', 'pytorch').
        """
        self.framework = framework

    def predict(self, model, X):
        """
        Generic predict method to support multiple frameworks.
        :param model: The model to use for prediction.
        :param X: Features for prediction.
        :return: Predicted values.
        """
        if self.framework == "sklearn":
            return model.predict(X)
        elif self.framework == "tensorflow":
            return model.predict(X).flatten()  # TensorFlow models return numpy arrays
        elif self.framework == "pytorch":
            model.eval()  # Ensure the model is in evaluation mode
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X.values, dtype=torch.float32)  # Convert to PyTorch tensor if not already
                return model(X).detach().numpy().flatten()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def evaluate(self, model, X_test, y_test):
        """
        Evaluate the model on the test set and return evaluation metrics.
        :param model: The trained model to evaluate.
        :param X_test: Test features.
        :param y_test: True target values.
        :return: Dictionary of evaluation metrics.
        """
        # Predict using the generic predict method
        y_pred = self.predict(model, X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        #spearman_corr, _ = spearmanr(y_test, y_pred)


        # Return all metrics as a dictionary
        return {
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            #"Spearman's Rank Correlation": spearman_corr,
        }
