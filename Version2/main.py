import os
import sys
import pandas as pd

from Version2.src.data_wrangling.SyntheticDistanceCalculator import SyntheticDistanceCalculator
from config import Config
from src.Models.model_2.random_tree import RandomForestModel
from src.Models.model_1.feed_forward_nn import NeuralNetworkModel
from src.Models.model_base.ranked_based_sorting import RankBasedSortingModel
from src.data_wrangling.data_processing import DataProcessor
from src.utils.create_target_for_learning import CreateTarget

num_samples = 1000
def build_test_frame(avalable_spots, walking_coords, driving_coords):
    ApiHold = SyntheticDistanceCalculator()

    test_frame = avalable_spots[['Lat', 'Long']]
    test_frame = pd.concat([test_frame, ApiHold.generate_test_trips(walking_coords, driving_coords, test_frame)], ignore_index=True)

    print(test_frame.head)
    return test_frame

def main():
    """
    Main function to prepare data, train models, and evaluate results.
    """
    walk_tupple = ((40.748817, -73.985428),(40.712743, -74.013379),(40.758896, -73.985130))
    drive_tupple = ((41.3129, -73.9884),(40.9168, -74.1817), (40.5795, -74.1441))
    # Step 1: Load data from CSV
    print("Loading data from CSV...")
    data_processor = DataProcessor(Config.CSV_PATH)

    # Step 1: Remove rows where 'Site_Reque' is "Renewal"
    final_cleaned_data = data_processor.data[data_processor.data['Site_Reque'] != 'Renewal']

    # Step 2: Drop columns that are entirely NaN
    final_cleaned_data = final_cleaned_data.dropna(axis=1, how='all')

    # Step 3: Remove rows with no data (entirely NaN rows)
    final_cleaned_data = final_cleaned_data.dropna(how='all')

    # Step 4: Remove rows outside newyork city
    final_cleaned_data = final_cleaned_data[
        (final_cleaned_data['Lat'].between(30, 70)) &
        (final_cleaned_data['Long'].between(-100, -60))
        ]

    # Step 5: Reset index for the cleaned dataset
    final_cleaned_data = final_cleaned_data.reset_index(drop=True)
    # File path for the processed dataset
    output_csv_path = "journey_dataset.csv"
    test_frame_path = "test_frame.csv"
    # Check if the file exists
    if os.path.exists(output_csv_path):
        print(f"{output_csv_path} exists. Loading the data...")
        test_data = pd.read_csv(output_csv_path)
    else:
        # creates test data if its not already there
        print(f"{output_csv_path} does not exist. Generating the dataset...")
        test_data = data_processor.generate_synthetic_trips(num_samples)
        data_processor.create_csv_from_dataframe(test_data, output_csv_path)
    # Check if the file exists
    # test_frame = build_test_frame(final_cleaned_data, walk_tupple[0], drive_tupple[0]) for testing


    print("Test Dataset:")
    print(test_data.head())
    # Create target score creator
    target_score_creator = CreateTarget()

    # Step 5: Provide a choice to the user
    print("\nChoose the model to run:")
    print("1. Ranked-Based Sorting Model")
    print("2. Neural Network Model")
    print("3. Random Forest Model")
    choice = input("Enter your choice (1,2 or 3): ").strip()

    if choice == "1":
        # Initialize the rank-based sorting model
        rank_model = RankBasedSortingModel()

        # Rank the data based on predefined criteria
        ranked_data = rank_model.rank_data(
            data_frame=test_data,
            sort_columns=['walking_distance', 'driving_distance', 'total_cost'],
            ascending_order=[True, True, True]  # Smaller values are better
        )
        print("Ranked Data:")
        print(ranked_data)

        # Test the model
        correlations = rank_model.test_model()

        # Visualize ranking results
        rank_model.visualize_ranking()

        # Visualize feature relationships with suitability score
        rank_model.visualize_features()

        # Get the top-2 ranked rows
        top_k_data = rank_model.get_top_k(k=2)
        print("Top-2 Ranked Data:")
        print(top_k_data)

    elif choice == "2":
        nn_model = NeuralNetworkModel(input_size=5, hidden_size=64, learning_rate=0.001)

        # Train
        nn_model.train(test_data, feature_columns=['walking_distance', 'walking_time', 'driving_distance', 'driving_time',
                                              'total_cost'], target_column='suitability_score')

        # Evaluate
        metrics = nn_model.evaluate()

        # Cross-Validation
        fold_metrics = nn_model.cross_validate(test_data,
                                               feature_columns=['walking_distance', 'walking_time', 'driving_distance',
                                                                'driving_time', 'total_cost'],
                                               target_column='suitability_score')

        # Visualize Errors
        nn_model.visualize_errors()
    elif choice == "3":
        print("\nRunning Random Tree...")
        rf_model = RandomForestModel(n_estimators=100, random_state=42)
        rf_model.train(
            data_frame=test_data,
            feature_columns=['walking_distance', 'walking_time', 'driving_distance', 'driving_time', 'total_cost'],
            target_column='suitability_score'
        )
        evaluation_metrics = rf_model.evaluate()
        print("Evaluation Metrics:", evaluation_metrics)

        feature_importances = rf_model.feature_importance()

        print("Feature Importances:", feature_importances)
        rf_model.residual_plot()
    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nProcess complete.")


if __name__ == "__main__":
    main()