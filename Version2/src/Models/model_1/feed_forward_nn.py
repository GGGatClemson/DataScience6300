import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ...utils.evaluator import Evaluator
from ...utils.visualizations import Visualizer
from ...utils.inference import Inference
from ...data_wrangling.data_processing import DataProcessor


class JourneyDataset(Dataset):
    def __init__(self, X, y):
        """
        Custom PyTorch Dataset for journeys.
        :param X: Features as a PyTorch tensor.
        :param y: Targets as a PyTorch tensor.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetworkModel:
    class JourneyNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            """
            Define a simple neural network structure.
            """
            super(NeuralNetworkModel.JourneyNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    def __init__(self,config, dataProcessor):
        """
        Initialize the neural network model and utilities.
        """
        self.model = None
        self.data_loader = None
        self.config = config
        self.data_processor = dataProcessor
        self.evaluator = Evaluator(framework="pytorch")
        self.visualizer = Visualizer(features=self.config.FEATURES, framework="pytorch")
        self.inference = Inference(num_samples=self.config.NUM_SAMPLES)

    def prepare_data(self, dataframe):
        """
        Prepare data for training and testing.
        :param dataframe: Input dataframe.
        """
        X = self.config.FEATURES
        y = self.config.TARGET

        # Split data
        X_train, X_test, y_train, y_test = self.data_processor.split_data(dataframe,
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE
        )

        # Prepare PyTorch datasets
        self.X_test = torch.tensor(X_test.values, dtype=torch.float32)
        self.y_test = torch.tensor(y_test.values, dtype=torch.float32)

        self.data_loader = DataLoader(
            JourneyDataset(torch.tensor(X_train.values, dtype=torch.float32),
                           torch.tensor(y_train.values, dtype=torch.float32)),
            batch_size=self.config.BATCH_SIZE, shuffle=True
        )

    def train(self):
        """
        Train the neural network model.
        """
        input_size = len(self.config.FEATURES)
        hidden_size = self.config.HIDDEN_SIZE
        output_size = 1

        # Initialize the model
        self.model = NeuralNetworkModel.JourneyNet(input_size, hidden_size, output_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            for X_batch, y_batch in self.data_loader:
                optimizer.zero_grad()  # Zero gradients
                outputs = self.model(X_batch).squeeze()  # Forward pass
                loss = criterion(outputs, y_batch)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.config.NUM_EPOCHS}], Loss: {loss.item():.4f}")

        print("Neural Network Model Training Complete.")

    def evaluate(self):
        """
        Evaluate the model using the evaluator class.
        """
        metrics = self.evaluator.evaluate(self.model, self.X_test, self.y_test)
        print("Evaluation Metrics:")
        print(metrics)

    def infer(self):
        """
        Perform inference and display top-k recommendations.
        """
        top_k_data = self.inference.get_top_k_recommendations_for(self.model, 10)
        print("Top-K Recommendations:")
        print(top_k_data)

    def visualize(self):
        """
        Visualize results using the visualizer class.
        """
        #y_pred = self.model(self.X_test).detach().numpy()
        #self.visualizer.plot_results(self.model, self.X_test, self.y_test, y_pred)
        self.visualizer.plot_results(self.model, self.X_test)

    def train_test_visualize(self, dataframe):
        """
        Train, test, perform inference, and visualize results for the neural network.
        :param dataframe: Input dataframe.
        """
        # Step 1: Prepare data
        self.prepare_data(dataframe)

        # Step 2: Train the model
        self.train()

        # Step 3: Evaluate the model
        self.evaluate()

        # Step 4: Perform inference
        self.infer()

        # Step 5: Visualize results
        self.visualize()

    def get_model(self):
        """
        Return the trained model.
        """
        return self.model
