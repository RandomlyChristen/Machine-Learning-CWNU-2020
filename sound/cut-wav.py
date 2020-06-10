import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

SAMPLING_RATE = 44100

#exit(0)

wav = 'test_ambulance/Bang/snipped525411.wav'
(file_dir, file_id) = os.path.split(wav)
print("file_dir:", file_dir)
print("file_id:", file_id)

# original
y, sr = librosa.load(wav, sr=SAMPLING_RATE)

if sr != SAMPLING_RATE:
    print("ERROR : 샘플링 레이트가 44100 이 아님!!!!", file=sys.stderr)
    exit(1)

# 원본 데이터 플롯
# time = np.linspace(0, len(y)/sr, len(y)) # time axis
# fig, ax1 = plt.subplots() # plot
# ax1.plot(time, y, color = 'b', label='speech waveform')
# ax1.set_ylabel("Amplitude") # y 축
# ax1.set_xlabel("Time [s]") # x 축
# plt.title(file_id) # 제목
# plt.savefig(file_id+'.png')
# plt.show()

# cut half and save
start_time = int(input("1초동안 자를 시작시간 입력 : "))
startSound = start_time * sr
endSound = startSound + sr

if endSound >= len(y):
    print("범위가 데이터의 범위 밖에 있습니다. 시작시간을 확인하세요!!", file=sys.stderr)
    exit(1)

y2 = y[startSound:endSound]

if len(y2) != sr:
    print("1초로 자른 사이즈가 이상함. len=%d" % len(y2), file=sys.stderr)
    exit(1)

librosa.output.write_wav('Bang5.wav', y2, sr) # save half-cut file
print("성공!!")

# time2 = np.linspace(0, len(y2)/sr, len(y2))
# fig2, ax2 = plt.subplots()
# ax2.plot(time2, y2, color = 'b', label='speech waveform')
# ax1.set_ylabel("Amplitude") # y 축
# ax1.set_xlabel("Time [s]") # x 축
# plt.title('cut '+file_id)
# plt.savefig('cut_half '+file_id+'.png')
# plt.show()