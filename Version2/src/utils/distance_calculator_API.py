import openrouteservice
import pandas as pd

"""
use https://openrouteservice.org/services/ for info 
visit https://openrouteservice.org/ login to get free api.

for api the format is (longitude, latitude) in our case the -70s first , 40s second.
"""

class DistanceCalculator():
    def __init__(self):
        #  OpenRouteService API key
        self.client = openrouteservice.Client(key='replace with your api key here')
        self.stations_df = pd.read_csv('parkingLocations.csv')

    def calculate_distance_time_driving(self,start_coords, end_coords):
        try:
            # Request route details from OpenRouteService
            route = self.client.directions(coordinates=[start_coords, end_coords], profile='driving-car', format='geojson')

            # Extract distance (meters) and duration (seconds)
            distance = route['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert to kilometers
            duration = route['features'][0]['properties']['segments'][0]['duration'] / 60  # Convert to minutes

            return distance, duration

        except Exception as e:
            print(f"Error occurred: {e}")
            return None, None

    def calculate_distance_time_walking(self, start_coords, end_coords):
        try:
            # Request route details from OpenRouteService
            route = self.client.directions(coordinates=[start_coords, end_coords], profile='foot-walking', format='geojson')

            # Extract distance (meters) and duration (seconds)
            distance = route['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert to kilometers
            duration = route['features'][0]['properties']['segments'][0]['duration'] / 60  # Convert to minutes

            return distance, duration

        except Exception as e:
            print(f"Error occurred: {e}")
            return None, None

    def get_cordinates_from_csv(self,rowNum):

        # Get the row from the dataframe using the row number
        row = self.stations_df.iloc[rowNum]
        # Extract longitude and latitude
        longitude = row['Long']
        latitude = row['Lat']
        # Return coordinates as [longitude, latitude]
        return [float(longitude), float(latitude)]

