import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

CSV_READ = pd.read_csv('urban_sound/mfcc.csv')

X_before = CSV_READ.values[:, :-1]
y = CSV_READ.values[:, -1]

random_picked = np.random.choice(X_before.shape[1], 2, replace=False)
# random_picked = [860, 91]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_before[y == 1, random_picked[0]], X_before[y == 1, random_picked[1]],
            c='b', marker='o', label='SAFE', s=10)
plt.scatter(X_before[y == 0, random_picked[0]], X_before[y == 0, random_picked[1]],
            c='r', marker='o', label='DANGER', s=10)
plt.title('Original')
plt.legend()
plt.xlabel("x%d" % random_picked[0])
plt.ylabel("x%d" % random_picked[1])

plt.subplot(1, 2, 2)
X_after, y = RandomUnderSampler().fit_sample(X_before, y)
plt.scatter(X_after[y == 1, random_picked[0]], X_after[y == 1, random_picked[1]],
            c='b', marker='o', label='SAFE', s=10)
plt.scatter(X_after[y == 0, random_picked[0]], X_after[y == 0, random_picked[1]],
            c='r', marker='o', label='DANGER', s=10)
plt.title('Under Sample')
plt.legend()
plt.xlabel("x%d" % random_picked[0])
plt.ylabel("x%d" % random_picked[1])
plt.savefig('result_plot/random_under_sampled.png')
