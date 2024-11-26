from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Random Forest Regressor with default or user-defined parameters.
        :param n_estimators: Number of trees in the forest.
        :param random_state: Random state for reproducibility.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_names = None
        self.target_name = None

    def train(self, data_frame, feature_columns, target_column, test_size=0.2):
        """
        Train the Random Forest Regressor on the provided dataset.
        :param data_frame: DataFrame containing the data.
        :param feature_columns: List of feature column names.
        :param target_column: Name of the target column.
        :param test_size: Fraction of data to use for testing.
        """
        self.feature_names = feature_columns
        self.target_name = target_column

        # Split data into features (X) and target (y)
        X = data_frame[feature_columns]
        y = data_frame[target_column]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)

        # Train the model
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the Random Forest model using the test dataset.
        :return: Dictionary with evaluation metrics (MSE, RMSE, MAE, R2 score).
        """
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Print metrics
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def cross_validate(self, data_frame, cv=5):
        """
        Perform cross-validation on the model and return RMSE scores.
        :param data_frame: DataFrame containing the full dataset.
        :param cv: Number of folds for cross-validation.
        :return: List of RMSE scores for each fold.
        """
        X = data_frame[self.feature_names]
        y = data_frame[self.target_name]

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        print(f"Cross-Validation RMSE Scores: {cv_rmse}")
        print(f"Mean CV RMSE: {cv_rmse.mean():.4f}")

        return cv_rmse

    def residual_plot(self):
        """
        Plot the residuals (errors) of the model predictions.
        """
        # Predict on the test set
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred

        # Plot residuals
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)  # Zero error line
        plt.xlabel("Predicted Suitability Score")
        plt.ylabel("Residual (True - Predicted)")
        plt.title("Residual Plot")
        plt.show()

    def feature_importance(self):
        """
        Retrieve feature importance values.
        :return: Dictionary of features and their importance scores.
        """
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet. Train the model before accessing feature importance.")

        # Get feature importances
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def find_best_entry(self, data_frame):
        """
        Find and return the entry with the best suitability score from a given DataFrame.
        The DataFrame must include 'lat' and 'longitude' columns.
        :param data_frame: DataFrame containing the data to test. Must include 'lat' and 'longitude'.
        :return: Row with the highest predicted suitability score, including lat and longitude.
        """
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet. Train the model before using this function.")

        # Ensure 'lat' and 'longitude' columns are present
        required_columns = ['lat', 'longitude'] + self.feature_names
        for col in required_columns:
            if col not in data_frame.columns:
                raise ValueError(f"The input DataFrame must include the '{col}' column.")

        # Predict suitability scores for the dataset
        X = data_frame[self.feature_names]
        data_frame['Predicted_Score'] = self.model.predict(X)

        # Find the entry with the highest score
        best_entry = data_frame.loc[
            data_frame['Predicted_Score'].idxmax(), ['lat', 'longitude'] + self.feature_names + ['Predicted_Score']]
        print(f"Best Entry:\n{best_entry}")
        return best_entry