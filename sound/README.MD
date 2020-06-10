from Google-Research AudioSet
https://github.com/jim-schwoebel/download_audioset
download.py
크롤링 한 데이터 중 몇가지를 직접 듣고 1초를 슬라이스 하였음
    - d_sound : 위험한 소리
    - s_sound : 평범한 소리
    

mix-normalized-and-weighted.py
각 데이터에서 무작위로 300 개를 선택하여, Scaling 이후, 0 ~ 1 사이의 가중치를 주고, 높은 가중치의 소리로 레이블링
    - data/0 : 위험한 소리
    - data/1 : 평범한 리 