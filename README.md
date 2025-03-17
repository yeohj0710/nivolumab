# nivolumab

## 📂 프로젝트 구조

```
/nivolumab
├── /Data files  # Data files가 준비되어 있어야 함
├── processed_data.parquet  # 통합 데이터 파일 (/250315/data_process.ipynb 실행 시 생성됨)
├── /그 외 폴더들...
└── README.md
```

## 파일 설명

- /250315/data_process.ipynb
  - 1만 개의 데이터 파일에서 중복되는 값을 제거하고, 하나의 데이터 파일로 압축
- /250315/train_infer.py
  - 모델 학습 및 테스트 코드
  - train(t)
    - processed_data.parquet을 기반으로 모델 학습을 진행하는 모드
    - 매 epoch마다 갱신되어 저장되는 checkpoint.pth가 모델에 해당
  - infer(i)
    - 학습시킨 모델 checkpoint.pth로 테스트를 진행하는 코드
    - 입력받은 BW, EGFR, SEX, RAAS, BPS, amt, II에 대해 CP(t) 그래프 생성
- /250315/parquet_to_cp_graph.py
  - 1만 개의 데이터 파일 중 하나의 인덱스(0~9999 중 하나)를 입력해 해당 데이터의 그래프를 생성
  - 학습 모델이 생성한 CP(t) 그래프와 실제 CP(t) 그래프를 비교해 보기 위함
- /250315/predicted_concentration.png
  - 학습시킨 모델이 생성한 CP(t) 그래프 (infer 모드)
- /250315/real*concentration*{i}.png
  - i번 인덱스 데이터의 CP(t) 그래프 (i: 0~9999)

## 학습 모델 설명

- /models/2503171025/checkpoint.pth
  - CP length가 특정 길이 이상인 데이터만 선택 후, 필요 구간만큼 잘라서 학습에 사용하였음 (padding 사용하지 않음)
  - 약 430 epoch, loss 약 300
  - batch size 128, learning rate 3e-3 → 1e-7까지 조정함
