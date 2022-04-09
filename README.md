# Dilab_MerlotReserve_Torch


Merlot Reserve 를 PyTorch로 재구현하는 프로젝트입니다.

(원본 코드 : https://github.com/rowanz/merlot_reserve)

## 사전학습 사용방법

### 사전학습 데이터 전처리
```python
python pretrain/data/prepro_Youtube_data.py --data [Your YouTube data] --out_dir [Your out directory]
```

### 사전학습 실행
```python
python pretrain/script/run_pretrain.py --data [Your Preprocessed YouTube data] --out_dir [Your out directory]
```

## VCR 파인튜닝 사용방법

### VCR 데이터 전처리
```python
python finetune/vcr/run_fntn_vcr.py 
```

### VCR 파인튜닝
```python
python finetune/vcr/run_fntn_vcr.py 
```
