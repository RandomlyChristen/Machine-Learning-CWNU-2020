import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from joblib import load
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ESTIMATOR_PATH = 'models/reduce-svc/model-urban_sound-class_weight-85.4%.joblib'
LEARNING_CURVE_PATH = 'result_plot/reduce-svc/urban_sound-class_weight-best_estimator.png'

#######################################################################
#####################   X, y 값을 만들어 줘야함    #######################
#######################################################################

CSV_FILE_PATH = 'urban_sound/mfcc.csv'
PCA_N_COMPONENTS_RATE = 0.9

csv_read = pd.read_csv(CSV_FILE_PATH, header=None)
X = csv_read.values[:, :-1]
y = csv_read.values[:, -1]
pca = PCA(n_components=PCA_N_COMPONENTS_RATE)
X = pca.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

#######################################################################
#######################################################################

estimator = load(ESTIMATOR_PATH)
estimator = clone(estimator)

# TODO https://github.com/scikit-learn/scikit-learn/issues/4921 ?????
train_sizes, train_scores, test_scores = \
    learning_curve(estimator=estimator, X=X, y=y,
                   train_sizes=np.linspace(0.1, 1.0, 10), cv=10,
                   # shuffle=True, random_state=777,
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
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1.03])
plt.legend(loc='lower right')
plt.savefig(LEARNING_CURVE_PATH)
