from config import Config
from src.Models.model_base.ranked_based_sorting import RankedBasedSorting
from src.Models.model_1.feed_forward_nn import NeuralNetworkModel
from src.data_wrangling.data_processing import DataProcessor
from src.utils.create_target_for_learning import CreateTarget
import pandas as pd
import sys

def main():
    """
    Main function to prepare data, train models, and evaluate results.
    """
    # Step 1: Load data from CSV
    print("Loading data from CSV...")
    data_processor = DataProcessor(Config.CSV_PATH)

    # Step 2: Preprocess data
    print("Preprocessing data...")
    data_processor.remove_rows(9)

    # Step 3: Generate training current positions and create the merged dataset
    training_current_positions = data_processor.generate_dummy_current_positions(3, 1)
    merged_data = pd.DataFrame()

    for current_position in training_current_positions:
        print(f"Processing current position: {current_position}")
        data_processor.calculate_distance_all_journeys(current_position)
        merged_data = pd.concat([merged_data, data_processor.new_data], ignore_index=True)

    print("Merged Dataset:")
    print(merged_data.head())

    # Step 4: Save the processed dataset
    output_csv_path = "journey_dataset.csv"
    data_processor.create_csv_from_dataframe(merged_data, output_csv_path)

    target_score_creator = CreateTarget()

    # uncomment to Train and evaluate the Ranked-Based Sorting model
    # Step 5: Provide a choice to the user
    print("\nChoose the model to run:")
    print("1. Ranked-Based Sorting Model")
    print("2. Neural Network Model")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        # Run Ranked-Based Sorting Model
        print("\nRunning Ranked-Based Sorting Model...")

        # Create target score (rank-based)
        merged_data = target_score_creator.create_rank_based_target_scores(
            merged_data, sort_by_columns=['availability', 'walk_time', 'distance_km', 'walk_distance', 'duration_min']
        )

        # Train and visualize Ranked-Based Sorting Model
        rbs_model = RankedBasedSorting(data_processor,merged_data, Config)
        rbs_model.train_test_visualize()

    elif choice == "2":
        # Run Neural Network Model
        print("\nRunning Neural Network Model...")

        # Create composite target score
        weight_dict = {
            'distance_km': 0.4,
            'duration_min': 0.3,
            'walk_distance': 0.2,
            'walk_time': 0.1
        }
        merged_data = target_score_creator.create_composite_target_score(merged_data, weight_dict)

        # Train and visualize Neural Network Model
        nn_model = NeuralNetworkModel(Config, data_processor)
        nn_model.train_test_visualize(merged_data)

    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nProcess complete.")


if __name__ == "__main__":
    main()
