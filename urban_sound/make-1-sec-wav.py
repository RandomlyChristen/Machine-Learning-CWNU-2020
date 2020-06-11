import librosa
import sys

import os

index = 0

for path, _, files in os.walk('UrbanSound/data'):
    for file in files:
        if not file.endswith('.wav'):
            continue

        file = os.path.join(path, file)

        try:
            data, sample_rate = librosa.load(file, sr=None)

            if sample_rate != 44100:
                continue

            if len(data.shape) > 1:
                continue

            for i in range(0, len(data) - sample_rate, sample_rate):
                librosa.output.write_wav('data_out/%d.wav' % index, data[i:i+sample_rate], sample_rate)
                index += 1

        except:
            print("ERROR : %s" % file, file=sys.stderr)
