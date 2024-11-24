import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from Version1.SyntheticDistanceCalculator import SyntheticDistanceCalculator


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

    def calculate_distance_all_journeys(self,current_position):
        """
          Calculate distances and times from all starting stations to all destination stations.
          """
        num_rows = len(self.data)
        car_sharers = self.data['Carshare_C'].tolist()

        # Prepare a list to hold data for the new DataFrame
        new_data = []

        # Iterate over all possible destination stations
        for dest_index, dest_row in self.data.iterrows():
            dest_coords = (dest_row['Lat'], dest_row['Long'])
            dest_car_sharer = dest_row['Carshare_C']

            # Iterate over all possible starting stations
            for start_index, start_row in self.data.iterrows():
                if start_index == dest_index:
                    # Skip if the starting station is the same as the destination station
                    continue

                start_coords = (start_row['Lat'], start_row['Long'])
                start_car_sharer = start_row['Carshare_C']

                # Calculate driving distance and time from starting station to destination station
                distance, time = self.distanceCalc.calculate_distance_time(start_coords, dest_coords)

                # Calculate walking distance and time from current position to starting station
                walk_distance, walk_time = self.distanceCalc.calculate_distance_time(
                    current_position, start_coords, 'walking'
                )

                # Determine availability
                availability = 1 if dest_car_sharer == start_car_sharer else 0

                # Append the journey data to the list
                new_data.append({
                    'start_station_id': start_index,
                    'destination_station_id': dest_index,
                    'distance_km': distance,
                    'duration_min': time,
                    'company_name': start_car_sharer,
                    'availability': availability,
                    'walk_distance': walk_distance,
                    'walk_time': walk_time
                })

        # Create a new DataFrame from the list
        self.new_data = pd.DataFrame(new_data)
        print("Generated journey data for all origin-destination pairs.")

    def generate_dummy_current_positions(self, n, max_distance_km=2):
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



