import matplotlib.pyplot as plt
import numpy as np


def save_validation_accuracy_2d(grid, file, title='Validation accuracy'):
    key_1, key_2 = (key for key in grid.param_grid.keys())
    range_1, range_2 = (value for value in grid.param_grid.values())

    scores = grid.cv_results_['mean_test_score'].reshape(len(range_2), len(range_1))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=.95, bottom=.15, top=.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel(key_1)
    plt.ylabel(key_2)
    plt.colorbar()
    plt.xticks(np.arange(len(range_1)), range_1, rotation=45)
    plt.yticks(np.arange(len(range_2)), range_2)
    plt.title(title)
    plt.savefig(file)
