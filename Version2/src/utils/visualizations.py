import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.inspection import PartialDependenceDisplay


class Visualizer:
    def __init__(self, features, framework="sklearn"):
        """
        Initialize the Visualizer with feature names and framework type.
        :param features: List of feature names.
        :param framework: Framework used for the model ('sklearn', 'tensorflow', 'pytorch').
        """
        self.features = features
        self.framework = framework

    def predict(self, model, X_test):
        """
        Generic predict method to handle predictions across frameworks.
        :param model: Trained model.
        :param X_test: Test feature set (can be DataFrame, NumPy array, or PyTorch tensor).
        :return: Predicted values as a NumPy array.
        """
        if self.framework == "sklearn":
            return model.predict(X_test)
        elif self.framework == "tensorflow":
            return model.predict(X_test).flatten()  # TensorFlow models return numpy arrays
        elif self.framework == "pytorch":
            model.eval()  # Ensure the model is in evaluation mode
            with torch.no_grad():
                # Convert to PyTorch tensor if not already
                if not isinstance(X_test, torch.Tensor):
                    if hasattr(X_test, 'values'):  # Handle DataFrame
                        X_test = torch.tensor(X_test.values, dtype=torch.float32)
                    else:  # Assume NumPy array
                        X_test = torch.tensor(X_test, dtype=torch.float32)
                return model(X_test).squeeze().numpy()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def plot_feature_importance(self, model):
        """
        Plot feature importance for tree-based models.
        :param model: Trained model with feature_importances_ attribute.
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(8, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [self.features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.show()
        else:
            print("Feature importance is not available for this model.")

    def plot_actual_vs_predicted(self, y_test, y_pred):
        """
        Plot actual vs predicted values.
        :param y_test: True target values.
        :param y_pred: Predicted target values.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Target Score')
        plt.ylabel('Predicted Target Score')
        plt.title('Actual vs. Predicted Target Scores')
        plt.show()

    def plot_residuals(self, y_test, y_pred):
        """
        Plot residuals of predictions.
        :param y_test: True target values.
        :param y_pred: Predicted target values.
        """
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Target Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    def plot_prediction_distribution(self, y_pred):
        """
        Plot the distribution of predicted values.
        :param y_pred: Predicted target values.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(y_pred, kde=True)
        plt.xlabel('Predicted Target Score')
        plt.title('Distribution of Predicted Target Scores')
        plt.show()

    def plot_cumulative_gains(self, y_test, y_pred):
        """
        Plot the cumulative gains chart.
        :param y_test: True target values.
        :param y_pred: Predicted target values.
        """
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_test.values[sorted_indices]

        cumulative_gains = np.cumsum(y_true_sorted)
        cumulative_gains /= cumulative_gains[-1]

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(cumulative_gains) + 1), cumulative_gains, marker='o')
        plt.xlabel('Number of Journeys')
        plt.ylabel('Cumulative Gain')
        plt.title('Cumulative Gains Chart')
        plt.show()

    def plot_partial_dependence(self, model, X_test):
        """
        Plot partial dependence for tree-based models.
        :param model: Trained model.
        :param X_test: Test feature set.
        """
        if self.framework == "sklearn" and hasattr(model, "feature_importances_"):
            fig, ax = plt.subplots(figsize=(12, 8))
            PartialDependenceDisplay.from_estimator(model, X_test, self.features, ax=ax)
            plt.show()
        else:
            print("Partial dependence plots are not supported for this model or framework.")

    def plot_results(self, model, X_test, y_test=None):
        """
        Main function to generate all visualizations.
        :param model: Trained model.
        :param X_test: Test feature set.
        :param y_test: True target values (optional).
        """
        print("Generating predictions...")
        y_pred = self.predict(model, X_test)

        if y_test is not None:
            print("Plotting Actual vs Predicted...")
            self.plot_actual_vs_predicted(y_test, y_pred)

            print("Plotting Residuals...")
            self.plot_residuals(y_test, y_pred)

            print("Plotting Cumulative Gains...")
            self.plot_cumulative_gains(y_test, y_pred)

        print("Plotting Prediction Distribution...")
        self.plot_prediction_distribution(y_pred)

        print("Plotting Feature Importance...")
        self.plot_feature_importance(model)

        print("Plotting Partial Dependence (if applicable)...")
        self.plot_partial_dependence(model, X_test)
