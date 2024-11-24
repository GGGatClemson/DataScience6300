import math


class SyntheticDistanceCalculator:
    """
    A utility class for calculating distances between coordinates and estimating travel times.
    use this only so you dont exhaust your api free usage limit
    """

    # Constants
    EARTH_RADIUS_KM = 6371.0
    KM_TO_MILES = 0.621371

    # Average speeds in km/h for different modes of transportation
    TRAVEL_SPEEDS = {
        'walking': 5,  # Average walking speed
        'cycling': 20,  # Average cycling speed
        'driving': 60,  # Average driving speed in cities/mixed conditions
        'highway': 90,  # Average highway driving speed
        'train': 120,  # Average high-speed train
        'airplane': 800  # Average commercial airplane cruise speed
    }

    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points using Haversine formula.
        """
        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences in coordinates
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Calculate distance in kilometers
        return SyntheticDistanceCalculator.EARTH_RADIUS_KM * c

    @staticmethod
    def distance_in_kilometers(lat1, lon1, lat2, lon2):
        """
        Get distance in kilometers between two points.
        """
        return SyntheticDistanceCalculator.calculate_distance(lat1, lon1, lat2, lon2)

    @staticmethod
    def distance_in_miles(lat1, lon1, lat2, lon2):
        """
        Get distance in miles between two points.
        """
        km_distance = SyntheticDistanceCalculator.calculate_distance(lat1, lon1, lat2, lon2)
        return km_distance * SyntheticDistanceCalculator.KM_TO_MILES

    @staticmethod
    def estimate_travel_time(distance_km, mode='driving'):
        """
        Estimate travel time in minutes based on mode of transportation.

        Parameters:
        distance_km: Distance in kilometers
        mode: Mode of transportation ('walking', 'cycling', 'driving', 'highway', 'train', 'airplane')

        Returns:
        Estimated time in minutes
        """
        if mode not in SyntheticDistanceCalculator.TRAVEL_SPEEDS:
            raise ValueError(f"Unknown travel mode: {mode}")

        speed = SyntheticDistanceCalculator.TRAVEL_SPEEDS[mode]

        # Add additional time based on mode-specific factors
        base_time = (distance_km / speed) * 60  # Convert hours to minutes

        # Add mode-specific overhead times
        if mode == 'airplane':
            # Add 2 hours for airport procedures (security, boarding, etc.)
            return base_time + 120
        elif mode == 'train':
            # Add 30 minutes for station procedures
            return base_time + 30
        else:
            # Add 10% buffer time for traffic/rest stops for other modes
            return base_time * 2.1

    def calculate_distance_time(self,start_coords, destination_coords,mode='driving'):
        start_lat = start_coords[1]
        start_long = start_coords[0]
        destination_lat = destination_coords[1]
        destination_long = destination_coords[0]

        distance_km = self.distance_in_kilometers(start_lat, start_long, destination_lat, destination_long)
        duration = self.estimate_travel_time(distance_km, mode)

        return distance_km,duration





# Example usage
if __name__ == "__main__":
    # Example coordinates (New York and Los Angeles)
    ny_lat, ny_lon = 40.7128, -74.0060  # New York
    la_lat, la_lon = 34.0522, -118.2437  # Los Angeles

    # Calculate distances
    distance_km = SyntheticDistanceCalculator.distance_in_kilometers(ny_lat, ny_lon, la_lat, la_lon)
    distance_miles = SyntheticDistanceCalculator.distance_in_miles(ny_lat, ny_lon, la_lat, la_lon)

    # Calculate travel times for different modes
    driving_time = SyntheticDistanceCalculator.estimate_travel_time(distance_km, 'driving')
    flying_time = SyntheticDistanceCalculator.estimate_travel_time(distance_km, 'airplane')

    print(f"Distance: {distance_km:.2f} km")
    print(f"Distance: {distance_miles:.2f} miles")
    print(f"Estimated driving time: {driving_time:.0f} minutes")
    print(f"Estimated flying time: {flying_time:.0f} minutes")