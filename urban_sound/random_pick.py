import numpy as np
import os
import scipy.io.wavfile as wavfile


# Min-Max, 데이터의 최대와 최소를 지정하여 re-scale
def rescaling(data, min_scaled, max_scaled):
    data = data.astype(np.float64)
    point = (data - data.min()) / (data.max() - data.min())

    data = ((max_scaled - min_scaled) * point) + min_scaled

    return data.astype(np.int16)


wav_list = [file for file in os.listdir('data_out') if file.endswith('.wav')]
random_picked = [file for file in np.random.choice(wav_list, 5000, replace=False)]

for file in random_picked:
    sample_rate, data = wavfile.read('data_out/' + file)
    data = rescaling(data, -3000, 3000)
    wavfile.write('random_picked/' + file, sample_rate, data)
