from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from learning_curve import save_learning_curve
from cv_accuracy import save_validation_accuracy_2d

CSV_FILE_PATH = 'urban_sound/mfcc.csv'  # 어반 사운드로 재시


class ReduceImbalancedPredictor:
    """
    불균형 데이터에 대한 PCA-차원축소
    methods:
        data_fit_transform
            - 학습데이터 -> 스케일링 -> PCA

        data_transform
            - 테스트데이터 -> 스케일링 -> PCA
    """
    def __init__(self, n_components=.90, sampling=None, random_state=999):
        self._pca = PCA(n_components)
        self._scaler = StandardScaler()
        self._sampler = \
            RandomUnderSampler(random_state=random_state) if sampling is 'under' else \
            RandomOverSampler(random_state=random_state) if sampling is 'over' else \
            None
        self._model = None

    def data_fit_transform(self, X, y):
        _X = self._pca.fit_transform(self._scaler.fit_transform(X))

        if self._sampler is not None:
            return self._sampler.fit_sample(_X, y)

        return _X, y

    def data_transform(self, X):
        return self._pca.transform(self._scaler.transform(X))

    def data_inverse_transform(self, X_t):
        return self._scaler.inverse_transform(self._pca.inverse_transform(X_t))

    def set_model(self, model):
        self._model = model
        return self

    def predict(self, X):
        return self._model.predict(X)


"""
------------------------------------------------------------------------
--                          MAIN SCRIPT                               --
------------------------------------------------------------------------
"""
if __name__ == '__main__':
    # header=None 으로 열 인덱싱을 사용하지 않음
    csv_read = pd.read_csv(CSV_FILE_PATH, header=None)

    # data : (N, 1287) MFCC 처리된 1초동안의 음성데이터
    # target : (N,) 해당 음성에 대한 위험소리 또는 일반소리 0, 1
    X_origin = csv_read.values[:, :-1]
    y_origin = csv_read.values[:, -1].astype(np.int16)

    """
    두 클래스간 데이터 수가 일치하지 않음, 확률에 의존하는 영향을 배제하기 위해,
    
    1) 클래스 별 패널티 계수 차별화 휴리스틱 적용
       (The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.)
    
    2) 언더샘플링 - 우세한 클래스의 데이터를 무작위로 제거
    
    3) 오버샘플링 - 열등한 클래스의 데이터를 중복
    """
    parameter_cases = [
        # 1)
        dict(
            class_weight=dict(np.ndenumerate(
                compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_origin))),
            sampling=None
        ),

        # 2)
        dict(
            class_weight=None,
            sampling='under'
        ),

        # 3)
        dict(
            class_weight=None,
            sampling='over'
        )
    ]

    for clf_params in parameter_cases:
        predictor = ReduceImbalancedPredictor(n_components=0.9, sampling=clf_params['sampling'], random_state=999)
        X_local, y_local = predictor.data_fit_transform(X_origin, y_origin)

        # 아래 범위에 대한 그리드 서치를 진행
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)

        # 모든 그리드 서치의 밸리데이션은 K-Fold 로 진행
        cv = KFold(5, shuffle=True, random_state=777)

        # 밸런싱 케이스에 따라 Estimator 의 파라미터 지정 및 학습
        grid = GridSearchCV(SVC(class_weight=clf_params['class_weight']), param_grid=param_grid, cv=cv)
        grid.fit(X_local, y_local)

        best_case = 'weighted@%s-' % ('balanced' if clf_params['class_weight'] is not None else 'none') + \
            'C@%f-' % grid.best_params_['C'] + \
            'gamma@%f-' % grid.best_params_['gamma'] +\
            'score@%f' % grid.best_score_
        print(best_case)

        # 그리드 서치 교차 검증 결과 플롯
        save_learning_curve(grid.best_estimator_, X_local, y_local,
                            'result_plot/reduce-svc/learning-curve_%s.png' % best_case, random_state=999)
        save_validation_accuracy_2d(grid,
                                    'result_plot/reduce-svc/grid-validation_%s.png' % best_case)

        dump(predictor.set_model(grid.best_estimator_), 'models/reduce-svc/%s.joblib' % best_case)
