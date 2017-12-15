import matplotlib.pyplot as plt
import numpy as np

import GPy


class EngineeringModel:
    def __init__(self, models):
        self.models = models

    def evaluate(self, points, output_index):
        if len(np.shape(points)) == 1:
            points = np.reshape(points, (1, -1))
        return self.models[output_index].predict(points)[0]


def plot_model_2d(model, points, values):
    # plot 2d Gaussian process classification model
    positive_idx = values == 1

    model.plot(levels=40, resolution=80, plot_data=False, figsize=(6, 5))
    plt.plot(points[positive_idx, 0], points[positive_idx, 1], '.', markersize=10, label='Positive')
    plt.plot(points[~positive_idx, 0], points[~positive_idx, 1], '.', markersize=10, label='Negative')
    plt.legend()
    plt.show()


def plot_model(points, values, kernel):
    model = GPy.models.GPRegression(points, values, kernel)
    model.optimize()
    print(model)

    test_points = np.linspace(1948, 1964, 400).reshape(-1, 1)
    prediction_mean, prediction_var = model.predict(test_points)
    prediction_std = np.sqrt(prediction_var).ravel()
    prediction_mean = prediction_mean.ravel()

    plt.figure(figsize=(5, 3))
    plt.plot(points, values, '.', label='Training data')
    plt.plot(test_points, prediction_mean, label='Prediction')
    plt.fill_between(test_points.ravel(), prediction_mean - prediction_std,
                     prediction_mean + prediction_std, alpha=0.3, label='Confidence')
    plt.legend()
