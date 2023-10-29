from sklearn.base import BaseEstimator, TransformerMixin
import pandas

class ControlExperimentalDifferencer(BaseEstimator, TransformerMixin):
    def __init__(self, control_group_size, experiment_group_size):
        self.control_group_size = control_group_size
        self.experiment_group_size = experiment_group_size

    def fit(self, X, y=None):
        # No fitting necessary, return self
        return self

    def transform(self, X):
        # Assuming X is a DataFrame
        df = pandas.DataFrame()
        for i in range(self.experiment_group_size):
            for j in range(self.control_group_size):
                df[f"t_c{j}x{i}"] = X[f"control{j+1}"] - X[f"experimental{i+1}"]
        return df

