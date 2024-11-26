import numpy as np
import pandas as pd
import random
import time as t
from .SyntheticDistanceCalculator import SyntheticDistanceCalculator
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, csv_file_path):
        # Step 1: Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csv_file_path)
        print("Initial Data Loaded:")
        print(self.data.head())  # Print a few rows to verify


        self.distanceCalc = SyntheticDistanceCalculator()  # replace this with API to get real Data


    def remove_rows(self, row_indices):
        """
        Remove specified rows from the DataFrame.
        :param row_indices: A list of row indices to be removed.
        """
        self.data = self.data.drop(row_indices, axis=0).reset_index(drop=True)
        print(f"Rows {row_indices} removed. Current DataFrame:")
        print(self.data.head())  # Print a few rows to verify

    def split_data(self,data_frame,features, target, test_size=0.2, random_state=42):
        X = data_frame[features]
        y = data_frame[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def create_csv_from_dataframe(self, dataframe, output_csv_file):
        """
        Create a CSV file from the given DataFrame.
        :param dataframe: The DataFrame to write to CSV.
        :param output_csv_file: The path of the output CSV file.
        """
        dataframe.to_csv(output_csv_file, index=False)
        print(f"DataFrame has been successfully saved to {output_csv_file}")

    def generate_synthetic_trips(self, num_samples):
        # API only lets 2000 calls per day, so using it for learning is an issue but for testing the model its fine
        # Random walking distances (0.1 km to 1 km)
        walking_distances = np.random.uniform(0.1, 1, num_samples)
        # Assume average walking speed: 5 km/h also adds some random difficulty so its not straight to distance
        walking_times = walking_distances * 12 * random.uniform(.9,1.1) # Assume average walking speed: 5 km/h

        # Random driving distances (1 km to 50 km)
        driving_distances = np.random.uniform(1, 50, num_samples)
        driving_times = driving_distances * 2  # Assume average driving speed: 30 km/h

        # Costs created from drive time and distance
        daily_cost = 87.5
        total_cost = np.zeros(num_samples)
        for index, trip in enumerate(driving_times):
            trip_hours = trip[1]/60
            if  trip_hours > 7:
                total_cost[index] = daily_cost + np.maximum(0, (driving_distances[index] - 200)) * 0.67
            else:
                hourly_cost = random.uniform(12.5 * trip_hours, 87.5)
                total_cost[index] = hourly_cost + np.maximum(0, (driving_distances[index] - 200)) * 0.67


        # Suitability scores (these weights would be set by user)
        # driving time is ignored bc distance is effective and time at the location and drive time are in total cost
        suitability_scores = (
            10 / walking_distances +
            10 / walking_times +
            10 / driving_distances -
            total_cost / 50
        )

        # Combine into a DataFrame
        data = pd.DataFrame({
            'walking_distance': walking_distances,
            'walking_time': walking_times,
            'driving_distance': driving_distances,
            'driving_time': driving_times,
            'total_cost': total_cost,
            'suitability_score': suitability_scores
        })

        return data


    def generate_new_positions(self, n, max_distance_km=2):
        """
        Generate 'n' dummy current positions within a walking distance of parking locations.
        :param n: Number of dummy current positions to generate.
        :param max_distance_km: Maximum distance in kilometers from parking locations.
        :return: A list of dummy current positions as (latitude, longitude) tuples.
        """
        # Convert max distance in kilometers to degrees (~1 km ~ 0.009 degrees)
        max_offset = max_distance_km * 0.009

        # Get the latitude and longitude columns
        latitudes = self.data['Lat']
        longitudes = self.data['Long']

        # Generate dummy positions
        dummy_positions = []
        for _ in range(n):
            # Randomly select a reference parking station
            idx = np.random.choice(len(self.data))
            base_lat = latitudes.iloc[idx]
            base_long = longitudes.iloc[idx]

            # Generate random offsets within the max_distance range
            offset_lat = np.random.uniform(-max_offset, max_offset)
            offset_long = np.random.uniform(-max_offset, max_offset)

            # Create a new position by adding offsets
            dummy_lat = base_lat + offset_lat
            dummy_long = base_long + offset_long

            # Add the new position to the list
            dummy_positions.append((dummy_lat, dummy_long))

        print(f"Generated {n} dummy current positions.")
        return dummy_positions



