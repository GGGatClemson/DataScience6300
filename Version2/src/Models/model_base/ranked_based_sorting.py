from sklearn.ensemble import RandomForestRegressor

from ...data_wrangling.data_processing import DataProcessor
from ...utils.evaluator import Evaluator
from ...utils.inference import Inference
from ...utils.visualizations import Visualizer

class RankedBasedSorting:
    def __init__(self, dataProcessor,dataframe,config):
        """
        Initialize the RankedBasedSorting class.
        :param dataframe: DataFrame containing input data.
        """
        self.data = dataframe
        self.model = None
        self.config = config

        # Utility classes
        self.data_processor = dataProcessor
        self.evaluator = Evaluator()
        self.inference = Inference(num_samples=100)
        self.visualizer = Visualizer(features=self.config.FEATURES, framework="sklearn")


    def prepare_data(self):
        """
        Split data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_processor.split_data(self.data,
            self.config.FEATURES,
            self.config.TARGET,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )
    def train_test_visualize(self):

        # Prepare data
        self.prepare_data()

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")
        metrics = self.evaluator.evaluate(self.model, self.X_test,self. y_test)
        print(metrics)
        self.inference.get_top_k_recommendations_for(self.model,5)
        #self.visualizer.plot_results(self.model,self.X_test,self.y_test,self.y_train)
        self.visualizer.plot_results(self.model, self.X_test,self.y_test)


    def get_model(self):
        """
        Return the trained model.
        :return: Trained model.
        """
        return self.model




