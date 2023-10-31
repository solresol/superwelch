from sklearn.base import BaseEstimator, TransformerMixin
import pandas

class ControlExperimentalDifferencer(BaseEstimator, TransformerMixin):
    def __init__(self, control_group_size, experiment_group_size, include_original_values=False):
        self.control_group_size = control_group_size
        self.experiment_group_size = experiment_group_size
        self.include_original_values = include_original_values

    def fit(self, X, y=None):
        # No fitting necessary, return self
        return self

    def transform(self, X):
        # Really should sort the values here
        df = pandas.DataFrame()
        for i in range(self.experiment_group_size):
            if self.include_original_values:
                df[f"x{i+1}"] = X[f"experimental{i+1}"]
            for j in range(self.control_group_size):
                df[f"t_c{j+1}x{i+1}"] = X[f"control{j+1}"] - X[f"experimental{i+1}"]
        if self.include_original_values:
            for j in range(self.control_group_size):
                df[f"c{j+1}"] = X[f"control{j+1}"]
        return df

