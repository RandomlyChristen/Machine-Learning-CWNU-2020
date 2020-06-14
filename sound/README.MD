# 1차 데이터 수집
[Google AudioSet](https://research.google.com/audioset/) 에는 2,084,320개의 소리가 527개의 Human-Labeled되어 있다.

위 데이터셋은, YouTube로부터 해당 레이블의 소리가 영상속에서 발생하는 시간과 끝나는 시간을 API로 제공한다.  

## d_sound, s_sound
`youtube-dl`로 크롤링 된, 위험 또는 안전 소리에 대한 10초 동안의 소리가 각각 레이블 되어 들어있다. 

## data
`cut-wav.py`로 절단되고, `mix-normalized-and-weighted.py`로 
정규화 및 가중치 합이 적용된 데이터가 각각 0과 1로 레이블 되어 들어있다. 

## download.py
이 소스코드의 라이선스는 [download_audioset](https://github.com/jim-schwoebel/download_audioset) 의 오픈 소스 라이선스를 따른다.

`labels.xlsx` 에서 분류한 위험소리를 직접 듣고 분류하기 위해 아래 코드를 추가함.
```python
d_sound = ['/m/01b_21', '/m/05tny_', '/m/0ghcn6', '/m/0cdnk', '/m/078jl', '/t/dd00036', '/t/dd00037',
           '/m/02_41', '/m/0k4j', '/m/01bjv', '/m/04qvtq', '/m/012n7d', '/m/012ndj', '/m/0284vy3',
           '/m/02x984l', '/m/03l9g', '/m/01d380', '/m/0g6b5', '/m/07qnq_y', '/m/07pws3f', '/m/07pjjrj',
           '/m/07pc8lb', '/m/04_sv', '/m/07qb_dv', '/m/07qv4k0', '/m/07plct2']

s_sound = ['/t/dd00129', '/m/015p6', '/m/0ytgt', '/m/01h8n0', '/m/01j3sz', '/t/dd00001', '/m/03m9d0z',
           '/m/07pggtn', '/m/07pbtc8',  '/m/0838f', '/m/0j6m2', '/m/04k94']
...
        if clabels != [] and d_sound.count(ylabels[i]) > 0:

            # change to the right directory
            newdir = defaultdir2 + clabels[0] + '\\'
...
```

## cut-wav.py
모든 소리 데이터에서 데이터로서 유효한 소리를 사람이 직접 판단하고 1초간 추출하여 사용하기위한 프로그램. 

## mix-normalized-and-weighted.py
1초로 절단된 wav파일의 PCM 데이터의 최대-최소 정규화를 적용하고, 무작위로 생성된 가중치에 따라,
서로 합해진 후, 높은 가중치를 가진 성분으로 레이블링하는 프로그램. 
```python
# Min-Max, 데이터의 최대와 최소를 지정하여 re-scale
def rescaling(data, min_scaled, max_scaled):
    data = data.astype(np.float64)
    point = (data - data.min()) / (data.max() - data.min())

    data = ((max_scaled - min_scaled) * point) + min_scaled

    return data.astype(np.int16)
...
    # 25% 이상 위험한 소리가 섞여있으면, 위험한 소리로 분류
    label = 0 if weight[0] > 0.25 else 1
...
```
