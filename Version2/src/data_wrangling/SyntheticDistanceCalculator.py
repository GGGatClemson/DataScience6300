import random
import time

import numpy as np
import openrouteservice
import pandas as pd
from dotenv import load_dotenv
import os


VALID_PROFILES = {
    "driving-car",
    "driving-hgv",
    "cycling-regular",
    "cycling-road",
    "cycling-mountain",
    "cycling-electric",
    'foot-walking',
    "foot-hiking",
}

class SyntheticDistanceCalculator:
    def __init__(self):
        """
        Initialize the SyntheticDistanceCalculator with the ORS API key.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve the API key from the environment
        api_key = os.getenv("ORS_API_KEY")
        if not api_key:
            raise ValueError("ORS_API_KEY is not set in the .env file")

        # Initialize the ORS client
        self.client = openrouteservice.Client(key=api_key)


    def calculate_distance_time(self, start_coords, end_coords, profile='driving-car'):
        """
        Calculate distance and time using the OpenRouteService API.
        :param start_coords: Tuple (latitude, longitude) of the start location.
        :param end_coords: Tuple (latitude, longitude) of the end location.
        :param profile: Transport mode, e.g., 'driving-car', 'foot-walking', etc.
        :return: Tuple (distance in km, duration in minutes).
        """
        if profile not in VALID_PROFILES:
            raise ValueError(f"Invalid profile '{profile}'. Must be one of {VALID_PROFILES}")
        try:
            # API request to get route details
            response = self.client.directions(
                coordinates=[start_coords[::-1], end_coords[::-1]],  # ORS requires (lon, lat)
                profile=profile,
                format='json'
            )
            # Extract distance (in meters) and duration (in seconds)
            distance_m = response['routes'][0]['summary']['distance']
            duration_s = response['routes'][0]['summary']['duration']

            # Convert to kilometers and minutes
            distance_km = distance_m / 1000
            duration_min = duration_s / 60
            return distance_km, duration_min
        except openrouteservice.exceptions.ApiError as e:
            print(f"OpenRouteService API error: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None, None

    def generate_test_trips(self, start_coords, driving_coords, destinations_df):
        """
        Generate a DataFrame with synthetic trip data using destination coordinates.
        :param start_coords: Tuple (latitude, longitude) of the starting location.
        :param destinations_df: DataFrame with 'latitude' and 'longitude' columns for destinations.
        :return: DataFrame with synthetic trip data.
        """
        walking_distances = []
        walking_times = []
        driving_distances = []
        driving_times = []
        total_costs = []

        daily_cost = 87.5

        for index, row in destinations_df.iterrows():
            end_coords = (row['Lat'], row['Long'])

            # Get walking and driving distances and times
            walking_distance, walking_time = self.calculate_distance_time(
                start_coords, end_coords, profile='foot-walking'
            )
            time.sleep(1.5)
            driving_distance, driving_time = self.calculate_distance_time(
                end_coords, driving_coords, profile='driving-car'
            )
            time.sleep(1.5)
            driving_distance = driving_distance * 2
            driving_time = driving_time * 2 + random.uniform(1, 360)
            if walking_distance is None or driving_distance is None:
                # Skip this trip if API failed
                continue

            # Cost calculation
            trip_hours = driving_time / 60
            if trip_hours > 7:
                total_cost = daily_cost + max(0, (driving_distance - 200)) * 0.67
            else:
                hourly_cost = random.uniform(12.5 * trip_hours, 87.5)
                total_cost = hourly_cost + max(0, (driving_distance - 200)) * 0.67

            # Append results
            walking_distances.append(walking_distance)
            walking_times.append(walking_time)
            driving_distances.append(driving_distance)
            driving_times.append(driving_time)
            total_costs.append(total_cost)

        # Suitability scores
        suitability_scores = (
            10 / np.array(walking_distances) +
            10 / np.array(walking_times) +
            10 / np.array(driving_distances) -
            np.array(total_costs) / 50
        )

        # Combine into a DataFrame
        data = pd.DataFrame({
            'walking_distance': walking_distances,
            'walking_time': walking_times,
            'driving_distance': driving_distances,
            'driving_time': driving_times,
            'total_cost': total_costs
        })

        return data

    def generate_new_positions(self, data, n, max_distance_km=2):
        """
        Generate 'n' dummy current positions within a walking distance of parking locations.
        :param n: Number of dummy current positions to generate.
        :param max_distance_km: Maximum distance in kilometers from parking locations.
        :return: A list of dummy current positions as (latitude, longitude) tuples.
        """
        # Convert max distance in kilometers to degrees (~1 km ~ 0.009 degrees)
        max_offset = max_distance_km * 0.009

        # Get the latitude and longitude columns
        latitudes = data['Lat']
        longitudes = data['Long']

        # Generate dummy positions
        dummy_positions = []
        for _ in range(n):
            # Randomly select a reference parking station
            idx = np.random.choice(len(data))
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