class Config:
    """
    Configuration class for storing global constants and settings.
    """
    CSV_PATH ="../Data/december.csv"
    FEATURES = ['distance_km', 'duration_min', 'walk_distance', 'walk_time', 'availability']
    TARGET = 'target_score'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    NUM_SAMPLES = 100  # For inference
    N_ESTIMATORS = 100  # For RandomForest
    LEARNING_RATE = 0.01  # Learning rate for neural network
    BATCH_SIZE = 1  # Batch size for neural network
    NUM_EPOCHS = 2  # Number of epochs for neural network training
    HIDDEN_SIZE = 10  # Hidden layer size for neural network