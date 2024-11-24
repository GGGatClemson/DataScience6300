import pandas as pd
import numpy as np
import torch


class Inference:
    def __init__(self, num_samples, noise_level=0.2):
        """
        Initialize the JourneyRecommender.
        :param num_samples: Number of journey samples to generate.
        :param noise_level: Level of noise to add to the generated data (default: 10%).
        """
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.features = ['distance_km', 'duration_min', 'walk_distance', 'walk_time', 'availability']

    def generate_inference_data(self):
        """
        Generate synthetic data of num_samples journeys with random values.
        :return: A DataFrame of generated journeys.
        """

        new_journeys = pd.DataFrame({
            'distance_km': np.random.uniform(0.5, 50, self.num_samples),
            'duration_min': np.random.uniform(5, 120, self.num_samples),
            'walk_distance': np.random.uniform(0.1, 5, self.num_samples),
            'walk_time': np.random.uniform(2, 60, self.num_samples),
            'availability': np.random.randint(0, 2, self.num_samples)
        })

        # Add noise to the test data
        new_journeys['distance_km'] += np.random.normal(0, self.noise_level * new_journeys['distance_km'])
        new_journeys['duration_min'] += np.random.normal(0, self.noise_level * new_journeys['duration_min'])
        new_journeys['walk_distance'] += np.random.normal(0, self.noise_level * new_journeys['walk_distance'])
        new_journeys['walk_time'] += np.random.normal(0, self.noise_level * new_journeys['walk_time'])

        # Ensure no negative values after adding noise
        new_journeys = new_journeys.clip(lower=0)



        return new_journeys

    def make_inference(self, model, input_data):
        """
        Make predictions on the given input data.
        :param model: The trained model to use for inference.
        :param input_data: DataFrame containing journey features.
        :return: DataFrame with predicted scores added.
        """
        # Ensure the input data has the correct columns
        input_data = input_data[self.features]

        # Check if the model is PyTorch
        if isinstance(model, torch.nn.Module):
            # Convert input data to a PyTorch tensor
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

            # Set model to evaluation mode
            model.eval()

            # Perform inference
            with torch.no_grad():
                predictions = model(input_tensor).squeeze().numpy()

        else:
            # Assume scikit-learn or compatible model with a `predict` method
            predictions = model.predict(input_data)

        input_data['predicted_score'] = predictions

        return input_data

    def get_top_k_recommendations(self, model, input_data, k):
        """
        Get the top-k recommendations based on the predicted scores.
        :param model: The trained model to use for inference.
        :param input_data: DataFrame containing feature values for prediction.
        :param k: Number of top recommendations to return.
        :return: DataFrame of top-k recommendations.
        """
        predicted_data = self.make_inference(model, input_data)
        if predicted_data is not None:
            # Add new columns for total_distance and total_time
            predicted_data['total_distance'] = predicted_data['distance_km'] + predicted_data['walk_distance']
            predicted_data['total_time'] = predicted_data['duration_min'] + predicted_data['walk_time']

            # Sort by the predicted score in descending order (higher scores are better)
            top_k_data = predicted_data.sort_values(by='predicted_score', ascending=False).head(k)
            print(f"Top {k} Recommendations:")

            # Temporarily set pandas display options to show all columns
            with pd.option_context('display.max_columns', None, 'display.width', None):
                print(top_k_data[['predicted_score', 'total_distance', 'total_time'] + self.features])

            return top_k_data


    def display_top_k_recommendations(self, top_k_data):
        """
        Display the top-k recommendations in a clear format.
        :param top_k_data: DataFrame of top-k recommendations.
        """
        from IPython.display import display, HTML
        display(HTML(top_k_data.to_html(index=False)))

    def get_top_k_recommendations_for(self, model, k_num):
        """
        Generate synthetic data, perform inference using the given model, and display top-k recommendations.
        :param model: The trained model to use for inference.
        :param k_num: Number of top recommendations.
        """
        # Generate synthetic data
        infer_journey = self.generate_inference_data()

        # Get top-k recommendations
        top_k_recommendations = self.get_top_k_recommendations(model, infer_journey, k_num)


        # Display recommendations
        self.display_top_k_recommendations(top_k_recommendations)


