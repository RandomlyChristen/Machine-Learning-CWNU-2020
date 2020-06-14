from scipy.fftpack import dct
import scipy.io.wavfile as wav
import numpy as np
import math
import decimal
import os
import csv
import matplotlib.pyplot as plt
import librosa.display


def magspec(frames, NFFT):
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def powspec(frames, NFFT):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    win = winfunc(frame_len)
    frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    return frames * win


def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def fbank(signal, samplerate, winlen, winstep,
          nfilt, nfft, lowfreq, highfreq, preemph,
          winfunc):
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = np.sum(pspec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat, energy


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


# Convert Frequency to Mel Scale
def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.)


# Convert Mel Scale to Frequency Mel Scale은 Filter Bank를 나눌때 어떤 간격으로 나누어야 하는지 알려주며, 간격을 나누는 방법은 아래와 같다
def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mfcc(signal, samplerate=44100):
    nfft = calculate_nfft(samplerate, 0.025)
    print('nfft', nfft)
    # 0.025 (winlen: 분석 길이 0.025), 0.01 (winstep), nfilter (26)
    feat, energy = fbank(signal, samplerate, 0.025, 0.01, 26, nfft, 0, None, 0.97, lambda x: np.ones((x,)))
    feat = np.log(feat)
    # 로그가 취해진 Filter Bank 에너지에 DCT를 계산한다. 이유는 두가지 Filter Bank는 모두 Overlapping 되어 있기 때문에 Filter Bank
    # 에너지들 사이에 상관관계가 존재하기 때문이다. DCT는 에너지들 사이에 이러한 상관관계를 분리 해주는 역할을 하며, 따라서 Diagonal Covariance Matrice
    # 를 사용할 수 있게 된다.(HMM Classifier와 유사함)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :13]
    feat = lifter(feat)
    feat[:, 0] = np.log(energy)
    return feat


def lifter(cepstra, L=22):
    nframes, ncoeff = np.shape(cepstra)
    n = np.arange(ncoeff)
    lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
    return lift * cepstra


def mfcc_subplot(data, title='MFCC'):
    librosa.display.specshow(data, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()


if __name__ == '__main__':
    # LABELED_DATA_PATH = 'sound/data/'
    # RESULT_CSV_PATH = 'sound/mfcc.csv'
    # 어반 사운드로 재시도, 20200611 이수균
    LABELED_DATA_PATH = 'urban_sound/data/'
    RESULT_CSV_PATH = 'urban_sound/mfcc.csv'
    RESULT_PLOT_MFCC_d_sound = 'result_plot/MFCC/0'
    RESULT_PLOT_MFCC_s_sound = 'result_plot/MFCC/1'

    csv_list = []

    for label in os.listdir(LABELED_DATA_PATH):
        if not os.path.isdir(LABELED_DATA_PATH + label):  # 예외 처리, 디렉토리가 아니면 터짐
            continue

        for wav_filename in os.listdir(LABELED_DATA_PATH + label):
            print('label',label)
            if not wav_filename.endswith('.wav'):
                continue

            # TODO : 샘플 플로팅 대략 4~8개 할 것. 위치 result_plot/mfcc-sample.png
            # sample_rate, data = wav.read(LABELED_DATA_PATH + label + '/' + wav_filename)
            # wav_filename_split = os.path.splitext(wav_filename)[0]
            # print(data.shape)
            # if label == '0':
            #     plt.figure(figsize=(10, 4))
            #     librosa.display.specshow(mfcc(data, sample_rate), x_axis='time')
            #     plt.colorbar()
            #     plt.title('MFCC')
            #     plt.tight_layout()
            #     plt.savefig(RESULT_PLOT_MFCC_d_sound + '/' + wav_filename_split + '.png')
            # elif label == '1':
            #     plt.figure(figsize=(10, 4))
            #     librosa.display.specshow(mfcc(data, sample_rate), x_axis='time')
            #     plt.colorbar()
            #     plt.title('MFCC')
            #     plt.tight_layout()
            #     plt.savefig(RESULT_PLOT_MFCC_s_sound + '/' + wav_filename_split + '.png')

            sample_rate, data = wav.read(LABELED_DATA_PATH + label + '/' + wav_filename)
            mfcc_array = np.ravel(mfcc(data, sample_rate))  # 여기가 스케일러자리가 아님 스케일러 삭제
            mfcc_array = np.append(mfcc_array, [int(label)]).tolist()

            csv_list.append(mfcc_array)

    with open(RESULT_CSV_PATH, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)