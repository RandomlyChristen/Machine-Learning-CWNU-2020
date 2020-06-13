import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


def save_learning_curve(estimator, X, y, file, title='Number of training samples', shuffle=False, random_state=999):
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=estimator, X=X, y=y,
                       train_sizes=np.linspace(0.1, 1.0, 10), cv=10,
                       shuffle=shuffle, random_state=random_state,
                       n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5,
             label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', marker='s', linestyle='--', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xlabel(title)
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1.03])
    plt.legend(loc='lower right')
    plt.savefig(file)
