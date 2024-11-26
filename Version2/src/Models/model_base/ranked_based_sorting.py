import pandas as pd
import matplotlib.pyplot as plt

class RankBasedSortingModel:
    def __init__(self):
        """
        Initialize the rank-based sorting model.
        """
        self.data = None
        self.ranked_data = None

    def rank_data(self, data_frame, sort_columns, ascending_order=None):
        """
        Rank the data based on specified columns and normalize ranks into suitability scores.
        :param data_frame: DataFrame containing the data to be ranked.
        :param sort_columns: List of column names to sort by.
        :param ascending_order: List of booleans specifying sort order for each column.
                                True for ascending, False for descending. Defaults to ascending for all.
        :return: DataFrame with added 'rank' and 'suitability_score' columns.
        """
        if ascending_order is None:
            ascending_order = [True] * len(sort_columns)

        # Sort the DataFrame
        ranked_df = data_frame.sort_values(by=sort_columns, ascending=ascending_order).reset_index(drop=True)

        # Assign ranks (1 = best, higher values = worse)
        ranked_df['rank'] = ranked_df.index + 1

        # Normalize ranks to create suitability scores (1 = best score, 0 = worst score)
        max_rank = ranked_df['rank'].max()
        ranked_df['suitability_score'] = 1 - (ranked_df['rank'] / max_rank)

        self.data = data_frame
        self.ranked_data = ranked_df
        return ranked_df

    def test_model(self):
        """
        Test the model by comparing rank-based suitability scores against individual features.
        Provides a summary analysis of how rankings align with feature values.
        """
        if self.ranked_data is None:
            raise ValueError("No ranked data available. Run 'rank_data' first.")

        # Analyze correlation between suitability scores and key features
        correlations = self.ranked_data.corr()['suitability_score']

        print("Feature Correlations with Suitability Score:")
        print(correlations)
        return correlations

    def visualize_ranking(self):
        """
        Visualize the ranked data using a bar chart of suitability scores.
        """
        if self.ranked_data is None:
            raise ValueError("No ranked data available. Run 'rank_data' first.")

        # Plot suitability scores
        plt.figure(figsize=(10, 6))
        plt.bar(self.ranked_data.index, self.ranked_data['suitability_score'], color='blue', alpha=0.7)
        plt.xlabel("Ranked Entries")
        plt.ylabel("Suitability Score")
        plt.title("Suitability Scores of Ranked Data")
        plt.show()

    def visualize_features(self):
        """
        Visualize the relationship between features and suitability scores.
        """
        if self.ranked_data is None:
            raise ValueError("No ranked data available. Run 'rank_data' first.")

        # Scatter plot for each feature vs suitability score
        feature_columns = [col for col in self.ranked_data.columns if col not in ['rank', 'suitability_score']]

        for feature in feature_columns:
            plt.figure(figsize=(8, 5))
            plt.scatter(self.ranked_data[feature], self.ranked_data['suitability_score'], alpha=0.7)
            plt.xlabel(feature)
            plt.ylabel("Suitability Score")
            plt.title(f"Suitability Score vs {feature}")
            plt.show()

    def get_top_k(self, k):
        """
        Retrieve the top-k ranked rows.
        :param k: Number of top rows to retrieve.
        :return: DataFrame with top-k rows.
        """
        if self.ranked_data is None:
            raise ValueError("No ranked data available. Run 'rank_data' first.")
        return self.ranked_data.head(k)