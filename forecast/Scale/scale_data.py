import pandas as pd
import numpy as np
class Scale:
    def __init__(self, data, feature_range):
        self.data = data
        self.feature_range = feature_range
        self.data_min = data.min()
        self.data_max = data.max()
    def count_len(self):
        return self.data.shape[0]
    def min_max_scaler(self):
        data_range = self.data_max - self.data_min
        if data_range == 0:
            return self.data
        else:
            scale = (self.feature_range[1] - self.feature_range[0]) / data_range
            min_d = self.feature_range[0] - self.data_min * scale
            scaled_data = self.data * scale + min_d
            return scaled_data
    def inverse_min_max_scaler(self):
        scaled_data=self.min_max_scaler()
        range_values = self.data_max - self.data_min
        original_tensor = scaled_data * range_values + self.data_min
        return original_tensor
def inverse_min_max_scaler(normalized_tensor, min_values, max_values):
    range_values = max_values - min_values
    original_tensor = normalized_tensor * range_values + min_values
    return original_tensor