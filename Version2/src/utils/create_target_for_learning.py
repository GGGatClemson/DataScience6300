
class CreateTarget:
    def __init__(self):
        pass

    def create_composite_target_score(self, data_frame, weight_dict): # for neural network
        """
        Create a composite score for the dataset to be used as the target.
        :param weight_dict: Dictionary containing the weight of each feature.
        :return: DataFrame with an additional column 'target_score'.
        """
        # Ensure all weights add up to 1
        total_weight = sum(weight_dict.values())
        assert abs(total_weight - 1.0) < 1e-5, "The weights should sum up to 1"

        # Create a composite score column as the target
        data_frame['target_score'] = (
                weight_dict['distance_km'] * data_frame['distance_km'] +
                weight_dict['duration_min'] * data_frame['duration_min'] +
                weight_dict['walk_distance'] * data_frame['walk_distance'] +
                weight_dict['walk_time'] * data_frame['walk_time']
        )

        # Invert target score so that a lower score represents a better journey
        # Normalize it between 0 and 1 for easier training
        max_score = data_frame['target_score'].max()
        min_score = data_frame['target_score'].min()
        data_frame['target_score'] = 1 - ((data_frame['target_score'] - min_score) / (max_score - min_score))

        print("Data with Composite Target Scores:")
        print(data_frame.head())

        return data_frame

    def create_rank_based_target_scores(self, input_dataframe, sort_by_columns=None): #target for rank based random forest
        """
        Generate rank-based target scores for the provided DataFrame.
        :param input_dataframe: DataFrame to be ranked.
        :param sort_by_columns: List of column names in order of ranking priority.
        :return: DataFrame with rank-based scores added.
        """
        # If no sort columns are provided, use default columns for ranking
        if sort_by_columns is None:
            sort_by_columns = ['distance_km', 'duration_min', 'walk_distance']

        # Sort the dataframe by the columns provided
        ranked_df = input_dataframe.sort_values(by=sort_by_columns, ascending=[True] * len(sort_by_columns))

        # Assign ranks based on sorted values (1 = best, higher values = worse)
        ranked_df['rank'] = ranked_df.reset_index().index + 1

        # Create a target score based on the rank - the lower the rank, the better the score
        # Normalizing ranks to get scores between 0 and 1
        max_rank = ranked_df['rank'].max()
        ranked_df['target_score'] = 1 - (ranked_df['rank'] / max_rank)

        print("Data with Rank-Based Target Scores:")
        print(ranked_df.head())

        # Return the ranked DataFrame
        return ranked_df
