from mfcc import mfcc
from reduce_svc import ReduceImbalancedPredictor
from joblib import load
from six.moves import queue
import numpy as np
import pyaudio
import sys


class MicStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        super().__init__()

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()

        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer
        )

        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()

        self.closed = False
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, out_size):
        prev_chuck = np.empty(self._chunk, dtype=np.int16)

        while not self.closed:
            current_chunk = self._buff.get()
            if current_chunk is None:
                continue

            current_chunk = np.frombuffer(current_chunk, dtype=np.int16)
            appended = np.append(prev_chuck, current_chunk)

            if len(appended) >= out_size:
                yield appended[:out_size]
                prev_chuck = appended[self._chunk:]
            else:
                prev_chuck = appended


if __name__ == '__main__':
    predictor: ReduceImbalancedPredictor = load('weighted@none-C@10.000000-gamma@0.001000-score@0.969781.joblib')

    with MicStream(44100, 22050) as stream:
        audio_generator = stream.generator(out_size=44100)

        for x in audio_generator:
            data = np.ravel(mfcc(x, 44100)).reshape(1, 1287)
            data = predictor.data_transform(data)
            pred = predictor.predict(data)
            if pred[0] == 1:
                print('SAFE')
            else:
                print('DANGER', file=sys.stderr)



