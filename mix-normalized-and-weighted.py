from scipy.io import wavfile
import os
import numpy as np
import random

D_SOUND_PATH = 'sound/d_sound/'
S_SOUND_PATH = 'sound/s_sound/'
OUTPUT_PATH = 'sound/data/'

MINMAX = 30000


# Min-Max, 데이터의 최대와 최소를 지정하여 re-scale
def rescaling(data, min_scaled, max_scaled):
    data = data.astype(np.float64)
    point = (data - data.min()) / (data.max() - data.min())

    data = ((max_scaled - min_scaled) * point) + min_scaled

    return data.astype(np.int16)


d_sound_labels = os.listdir(D_SOUND_PATH)
s_sound_labels = os.listdir(S_SOUND_PATH)

# 위험한 소리 = 0, 일반적인 소리 = 1
d_sounds = [(D_SOUND_PATH + label_path + '/' + filename, 0)
            for label_path in os.listdir(D_SOUND_PATH) if label_path[0] != '.'
            for filename in os.listdir(D_SOUND_PATH + label_path) if os.path.splitext(filename)[1] == '.wav']

s_sounds = [(S_SOUND_PATH + label_path + '/' + filename, 1)
            for label_path in os.listdir(S_SOUND_PATH) for filename in os.listdir(S_SOUND_PATH + label_path)
            if os.path.splitext(filename)[1] == '.wav']

d_s_i = [0, 0]
for i, j in zip(np.random.choice(np.arange(len(d_sounds)), 300, replace=True),
                np.random.choice(np.arange(len(s_sounds)), 300, replace=True)):
    weight = random.randint(50, 100) / 100
    weight = [1 - weight, weight]

    d_path, d_label = d_sounds[i]
    s_path, s_label = s_sounds[j]

    sampling_rate_1, d_data = wavfile.read(d_path)
    sampling_rate_2, s_data = wavfile.read(s_path)

    if sampling_rate_1 != sampling_rate_2:  # 이건 두고볼 수 없음
        continue

    d_data, s_data = np.array(d_data), np.array(s_data)

    if d_data.ndim > 1:
        d_data = d_data[:, 0]

    if s_data.ndim > 1:
        s_data = s_data[:, 0]

    if len(d_data) != len(s_data):  # 가공 되지 않은 데이터임
        continue

    '''
    이하, 데이터는 반드시 1초, 반드시 44100
    '''

    d_data = (rescaling(d_data, -MINMAX, MINMAX) * weight[d_label]).astype(np.int16)
    s_data = (rescaling(s_data, -MINMAX, MINMAX) * weight[s_label]).astype(np.int16)

    # 25% 이상 위험한 소리가 섞여있으면, 위험한 소리로 분류
    label = 0 if weight[0] > 0.25 else 1

    wavfile.write("%s%d/%d.wav" % (OUTPUT_PATH, label, d_s_i[label]), sampling_rate_1, d_data + s_data)
    d_s_i[label] += 1
