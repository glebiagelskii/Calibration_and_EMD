from sklearn.linear_model import LinearRegression
import numpy as np


def fit_line(contrast_means, reference_masses):
    model = LinearRegression()
    contrast_means = contrast_means.reshape(-1, 1)
    model.fit(contrast_means, reference_masses)
    gradient = model.coef_[0]
