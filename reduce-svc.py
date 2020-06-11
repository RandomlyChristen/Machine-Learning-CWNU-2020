from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump
from imblearn.under_sampling import RandomUnderSampler


def plot_data(X, file, r, c, indexes):
    for i, index in enumerate(indexes):
        plt.subplot(r, c, i + 1)
        plt.imshow(X[index].reshape(99, 13), aspect='auto')

    plt.savefig(file)


# CSV_FILE_PATH = 'sound/mfcc.csv'
CSV_FILE_PATH = 'urban_sound/mfcc.csv'  # 어반 사운드로 재시
PCA_N_COMPONENTS_RATE = 0.9

# header=None 으로 열 인덱싱을 사용하지 않음
csv_read = pd.read_csv(CSV_FILE_PATH, header=None)

X = csv_read.values[:, :-1]
y = csv_read.values[:, -1]

# 두 클래스간 데이터 수가 일치하지 않음, 확률에 의존하는 영향을 배제하기 위해
# 클래스 간 데이터 수를 일치시킴, 언더샘플링 : 우세한 클래스의 데이터를 무작위로 제거
X, y = RandomUnderSampler(random_state=999).fit_sample(X, y)

before_pca = X.shape[1]

plot_data(X, 'result_plot/reduce-svc_before-pca.png', 2, 4, np.arange(8))

# PCA 적용
pca = PCA(n_components=PCA_N_COMPONENTS_RATE)
X = pca.fit_transform(X)
after_pca = X.shape[1]

plot_data(pca.inverse_transform(X),
          'result_plot/reduce-svc_pca-%d-comp.png' % pca.n_components_,
          2, 4, np.arange(8))

print("PCA 차원압축 : %d / %d" % (after_pca, before_pca))

'''
RBF Kernel SVC in GridSearch with K-Fold Cross-Validation,
'''
# X 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

cv = KFold(5, shuffle=True, random_state=777)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print('베스트 파라미터 : %s, 점수 %0.2f' % (grid.best_params_, grid.best_score_))

# 모델 출력
dump(scaler, 'models/reduce-svc_scaler.joblib')
dump(grid.best_estimator_, 'models/reduce-svc_model%d%%.joblib' % int(grid.best_score_ * 100))

# 그리드 서치 교차 검증 결과 플롯
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=.95, bottom=.15, top=.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.savefig('result_plot/reduce-svc_grid-result.png')
