from joblib import load
import pandas as pd
import numpy as np
from mfcc import mfcc_subplot
from reduce_svc import ReduceImbalancedPredictor
import matplotlib.pyplot as plt

MODEL_PATH = 'models/reduce-svc/weighted@none-C@10.000000-gamma@0.001000-score@0.969781.joblib'
CSV_FILE_PATH = 'urban_sound/mfcc.csv'

predictor: ReduceImbalancedPredictor = load(MODEL_PATH)

csv_read = pd.read_csv(CSV_FILE_PATH, header=None)
X_0 = csv_read.head().values[:, :-1]
y_0 = csv_read.head().values[:, -1].astype(np.int16)
X_1 = csv_read.tail().values[:, :-1]
y_1 = csv_read.tail().values[:, -1].astype(np.int16)

X_ap = np.append(X_0, X_1, axis=0)
y_ap = np.append(y_0, y_1)

for i in np.random.choice(len(y_ap), len(y_ap), replace=False):
    X, y = X_ap[i], y_ap[i]
    target_label = 'Label : DANGER' if y == 0 else 'Label : SAFE'

    X_tran = predictor.data_transform(X.reshape(1, -1))
    X_inv = predictor.data_inverse_transform(X_tran)

    pred = predictor.predict(X_tran)
    predict_label = 'Predict : DANGER' if pred[0] == 0 else 'Predict : SAFE'

    plt.figure(figsize=(10, 4))

    # 원본 데이터 출력
    plt.subplot(1, 2, 1)
    mfcc_subplot(X.reshape(99, 13), title=target_label)

    # 예측 데이터 출력
    plt.subplot(1, 2, 2)
    mfcc_subplot(X_inv.reshape(99, 13), title=predict_label)

    plt.savefig('result_plot/reduce-svc/predict_result/%d.png' % i)
