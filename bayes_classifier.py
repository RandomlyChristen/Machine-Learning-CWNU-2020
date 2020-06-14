from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import dump
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from learning_curve import save_learning_curve

CSV_FILE_PATH = 'urban_sound/mfcc.csv'  # 어반 사운드로 재시
MODEL_OUTPUT_PATH = 'models/model_urban_sound-class_weight_%.1f%%.joblib'
SCALER_OUTPUT_PATH = 'models/scaler_urban_sound-class_weight.joblib'
VALIDATION_RESULT_PATH = 'result_plot/urban_sound-bayesian_%s_%0.2f.png'
DATA_PLOT_PATH = 'result_plot/urban_sound-%s-bayesian.png'


def plot_data(X, file, r, c, indexes):
    for i, index in enumerate(indexes):
        plt.subplot(r, c, i + 1)
        plt.imshow(X[index].reshape(99, 13), aspect='auto')

    plt.savefig(file)


PCA_N_COMPONENTS_RATE = 0.9

# header=None 으로 열 인덱싱을 사용하지 않음
csv_read = pd.read_csv(CSV_FILE_PATH, header=None)

X = csv_read.values[:, :-1]
y = csv_read.values[:, -1]
y = y.astype(np.int16)

# 두 클래스간 데이터 수가 일치하지 않음, 확률에 의존하는 영향을 배제하기 위해,
# TODO-1 언더샘플링 : 우세한 클래스의 데이터를 무작위로 제거
X, y = RandomUnderSampler(random_state=999).fit_sample(X, y)

# TODO-2 오버샘플링 : 열등한 클래스의 데이터를 중복
# X, y = RandomOverSampler(random_state=999).fit_sample(X, y)

before_pca = X.shape[1]

plot_data(X, DATA_PLOT_PATH % 'BEFORE', 2, 4, np.arange(8))

# PCA 적용
pca = PCA(n_components=PCA_N_COMPONENTS_RATE)
X = pca.fit_transform(X)
after_pca = X.shape[1]

plot_data(pca.inverse_transform(X),
          DATA_PLOT_PATH % ('AFTER_comp:%d' % pca.n_components_),
          2, 4, np.arange(8))

print("PCA 차원압축 : %d / %d" % (after_pca, before_pca))

'''
RBF Kernel SVC in GridSearch with K-Fold Cross-Validation,
'''
# X 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Priors = np.logspace(-2, 10, 13)
Var_smoothing = np.logspace(-9, 3, 13)

param_grid = dict(var_smoothing=Var_smoothing)

skf = StratifiedKFold(n_splits=10)

nb2 = GridSearchCV(GaussianNB(), cv=skf, param_grid=param_grid)
nb2.fit(X, y)

print(nb2.best_params_['var_smoothing'])

print('베스트 파라미터 : %s, 점수 %0.2f' % (nb2.best_params_, nb2.best_score_))

save_learning_curve(nb2.best_estimator_, X, y,
                    'result_plot/bayes_classifier/learning-curve_UnderSamplering.png', random_state=999)

# 모델 출력
dump(nb2.best_estimator_, 'models/bayes_classifier/gaussianNB@UnderSamplering@score-%0.2f.joblib' % nb2.best_score_)
