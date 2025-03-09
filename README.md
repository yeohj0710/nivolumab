# nivolumab

## 📂 프로젝트 구조

```
/nivolumab
├── /Data files
├── processed_data.parquet  # 통합 환자 데이터 파일
├── /그 외 폴더들...
└── README.md
```

## 학습 모델 메모

- 2503081551
  - 최초로 학습 성공한 모델
  - 700 epoch 학습, loss 약 250 (수렴)
- 2503091447
  - batch size를 128로 줄여 일반화 수준 향상
  - GPU 2개 병렬로 모두 사용 (기존 코드는 1개만 사용)
  - 128 epoch 학습, loss 약 300 (아직 수렴 X)
- 2503100824
  - batch size 128, 100 epoch, loss 450
  - epoch 진행될수록 속도 저하됨 (원인 분석 중)