# 2차 데이터 수
[Urban Sound Dataset](https://urbansounddataset.weebly.com/)
에는 1,302개의 소리가 10개의 클래스로 레이블되어 있다.

위 데이터셋은 음원 자체를 직접 제공한다. 제공된 파일의 확장자는 m4a, mp3, wav 등으로 다양함.   

## UrbanSound
다운로드한 모든 음원 파일이 각각 레이블의 디렉토리로 저장되어 있다. 

## data_out
`make-1-sec-wav.py`로 추출된 1초의 wav파일들이 저장되어 있다. 

## random_picked
`random_pick.py`로 1초의 wav파일들을 무작위로 5000개 선정하여 저장했다.

## data
`random_picked`에 저장된 소리를 직접 들어보며, 레이블링한 결과가 저장되어 있다.

## make-1-sec-wav-py
다운로드된 데이터 중, wav파일만 다루고(다른 인코딩 방식에 대해 잘 모름) 모든 wav파일을 1초 단위로 추출한다.

## random_pick.py
직접듣고 레이블링 하기위해, 5000개의 데이터만 무작위로 선정하며, PCM 데이터를 정규화하여,
레이블링 시, 진폭이 영향을 미치는 것을 사전에 예방하였다. 

```python
# Min-Max, 데이터의 최대와 최소를 지정하여 re-scale
def rescaling(data, min_scaled, max_scaled):
    data = data.astype(np.float64)
    point = (data - data.min()) / (data.max() - data.min())

    data = ((max_scaled - min_scaled) * point) + min_scaled

    return data.astype(np.int16)
...
```
