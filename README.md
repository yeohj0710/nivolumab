# nivolumab

## 📂 프로젝트 구조

```
/nivolumab
├── /Data files  # Data files가 준비되어 있어야 함
├── processed_data.parquet  # 통합 데이터 파일 (/250315/data_process.ipynb 실행 시 생성됨)
├── /그 외 폴더들...
└── README.md
```

## 학습 모델 설명

아래는 3가지 코드 버전이 환자 정적 특성과 약물 주입 정보를 활용해 CP(t) 그래프를 예측하기 위해 어떤 방식으로 모델을 학습하는지에 대한 상세한 요약입니다.

---

## 코드 1 (250315): Neural ODE 기반 연속시간 모델

- **모델 구성 및 아이디어**  
  - **정적 인코딩:**  
    정적 특성 7개(`BW`, `EGFR`, `SEX`, `RAAS`, `BPS`, `amt`, `II`)를 입력받아, 간단한 인코더(Linear → ReLU)를 통해 저차원 임베딩을 생성합니다.  
  - **초기 상태 생성:**  
    인코딩된 정적 임베딩을 기반으로 초기 은닉 상태를 생성(추가 Linear 계층)합니다.
  - **미분방정식 (ODE) 함수:**  
    ODEFunction 모듈은 시간 t에서 상태를 미분하는 함수로,  
    - 현재 상태, 정적 임베딩, 그리고 **주입 효과**를 고려한 입력을 받아 네트워크(Linear → Tanh → Linear)로 미분값을 산출합니다.  
    - 주입 효과는 주입 dosis를 가우시안 커널(주입 시점과의 거리 기반)로 가중치 처리하여 반영합니다.
  - **수치적분 (odeint):**  
    `torchdiffeq`의 `odeint`를 사용해 미분방정식의 해(은닉 상태의 시간 변화)를 계산합니다.  
  - **출력 레이어:**  
    ODE 통과 후의 은닉 상태에 대해 `readout` 계층(Linear + Softplus)을 적용하여 CP 농도를 예측합니다.

- **데이터 및 학습 전략**  
  - **데이터셋:**  
    - Parquet 파일에서 데이터를 읽어 CP 시퀀스와 정적 특성을 함께 로드합니다.  
    - CP 시퀀스의 길이를 제한하거나 최소 길이를 충족하는 데이터를 필터링합니다.
  - **배치 처리:**  
    - 개별 시퀀스의 길이가 다르므로 `pad_sequence`를 사용하여 배치 단위로 패딩 처리하고, 배치마다 시간 축(t)도 재구성합니다.
  - **손실 및 최적화:**  
    - 예측된 CP 시퀀스와 실제 시퀀스 간의 차이를 MSE 손실로 계산하고, 패딩된 부분은 마스크를 적용해 제외합니다.
    - Adam 최적화 알고리즘과 기울기 클리핑을 사용해 학습을 진행합니다.
  - **분산 학습:**  
    - GPU가 여러 개인 환경에서는 분산 데이터 병렬 처리(`torch.distributed` 등)를 활용하여 학습을 진행합니다.
  - **추론:**  
    - 사용자로부터 정적 입력과 총 시간, 주입 관련 정보를 입력받아 모델 추론 후 결과를 CSV 및 그래프로 저장합니다.

> **요약:** 정적 특성을 통해 초기 상태를 정하고, Neural ODE를 통해 연속 시간상에서 약동학 동역학을 모델링하며 주입 효과를 직접 반영하는 방식으로 CP(t) 예측 문제를 해결합니다.

---

## 코드 2 (250408): MLP 기반 시계열 회귀 및 주입 효과 예측

- **모델 구성 및 아이디어**  
  - **데이터 전처리:**  
    - 각 환자 데이터 행(row)을 받아 CP 시퀀스를 **시계열 데이터 프레임**으로 변환합니다.  
    - 정적 특성은 각 시점에 동일하게 반복되며, CP 값은 첫 값(기저치)을 기준으로 보정(차감)합니다.
    - 특히, 주입 시점에 해당하는 위치(`time % II == 0`)에서 `amt` 값을 반영하여 주입 효과를 강조합니다.
  - **모델 구조 (MLPEstimate):**  
    - 입력 차원은 7이며, 다층 퍼셉트론(MLP) 구조로 구성됩니다.
    - 은닉층은 256 차원으로, 총 10개의 레이어와 SiLU 활성화 함수를 사용해 깊게 쌓습니다.
    - 마지막 레이어에서 1차원 CP 예측 값을 출력합니다.

- **데이터셋 구성 및 배치 처리**  
  - **TimeSeriesDataset:**  
    - 변환된 시계열 데이터 프레임 리스트를 기반으로, 각 샘플의 정규화를 수행하여 입력과 출력(다음 시점 CP 값)을 구성합니다.
  - **collate_fn:**  
    - 서로 다른 길이의 시계열을 배치별로 패딩하고,  
    - 두 가지 입력 세트를 만듭니다.  
      1. **일반 시퀀스 입력:** 전체 시계열에 대해 이전 CP와 정적 특성으로 다음 CP 값을 예측하는 손실을 계산.  
      2. **주입 관련 입력:** 5회 주입과 관련된 시점의 CP 및 정적 특성을 따로 추출하여 주입 후 CP 값 예측 손실을 추가로 계산.

- **학습 전략**  
  - **두 가지 손실:**  
    - 전체 시계열에 대한 MSE (마스크를 적용하여 패딩 부분 무시)  
    - 주입 시점 관련 추가 손실 (injection loss)  
    - 두 손실의 합을 총 손실로 계산해 역전파 수행.
  - **최적화:** Adam Optimizer를 사용하며, epoch마다 주기적으로 체크포인트를 저장합니다.
  - **추론:**  
    - 테스트 데이터셋에 대해 예측값과 실제 CP 값의 시계열을 비교하고, 특정 샘플에 대한 그래프를 저장하여 시각적으로 평가합니다.

> **요약:** MLP 모델을 통해 각 시점별 CP 예측을 수행하는 동시에, 주입 이벤트에 따른 효과를 명시적으로 학습하는 방식입니다. 데이터 전처리 단계에서 시계열 확장을 통해 정적 특성과 주입 정보를 시간 축에 맞게 반영하는 점이 특징입니다.

---

## 코드 3 (250413): 정적 입력 기반 시퀀스 생성 (Static-to-Sequence 모델)

- **모델 구성 및 아이디어**  
  - **데이터 전처리:**  
    - 각 환자 행을 받아 CP 시퀀스를 확장하고, 정적 특성(7개)은 첫 행 값만 사용합니다.
    - CP 시퀀스는 정규화 과정을 거쳐 모델의 입력(라벨)으로 사용됩니다.
  - **모델 아키텍처 (StaticToSequenceModel):**  
    - **정적 조건 네트워크:**  
      - 정적 입력을 latent 공간(예, 256 차원)으로 매핑하는 MLP를 사용합니다.
    - **Positional Embedding:**  
      - 학습 가능한 positional embedding (최대 길이 100)을 사용하며, 만약 시퀀스 길이가 이를 초과하면 동적 계산한 **sinusoidal positional encoding**을 활용합니다.
    - **디코더:**  
      - 1D Convolution 블록(Conv1d + ReLU)을 여러 층 사용하여, 입력 latent 벡터와 positional embedding의 합을 기반으로 최종 CP 시퀀스를 생성합니다.
    - **입력 결합:**  
      - 정적 임베딩은 전체 시퀀스 길이에 대해 반복되고, 여기에 positional embedding을 더해 순서 정보를 주입한 뒤 디코더로 전달됩니다.

- **데이터셋 구성 및 배치 처리**  
  - **TimeSeriesDataset:**  
    - 각 샘플에 대해 정적 벡터와 CP 시퀀스를 생성합니다.
  - **collate_fn:**  
    - 서로 다른 길이의 CP 시퀀스를 동일한 길이로 패딩하고, 패딩 마스크 및 실제 시퀀스 길이 정보를 함께 반환합니다.

- **학습 전략**  
  - **손실 함수:**  
    - 디코더가 출력한 CP 시퀀스와 실제 라벨 간의 MSE 손실(Mask 적용)을 계산하여 학습합니다.
  - **최적화:**  
    - Adam Optimizer를 사용하며, 주기적으로 체크포인트를 저장합니다.
  - **추론:**  
    - 테스트셋에서 예측한 시퀀스와 실제 CP 시퀀스를 시각화해 결과를 평가합니다.

> **요약:** 정적 특성만을 입력받아 전체 CP 시퀀스를 한 번에 생성하는 방식입니다. 조건부 생성(Conditional Generation) 구조를 채택하여, 정적 입력을 latent 벡터로 매핑하고 positional 정보와 결합한 후 Convolutional 디코더를 사용해 시퀀스 전체를 출력하는 점이 핵심입니다.

---

## 종합 비교

- **코드 1 (Neural ODE):**  
  - **연속시간 모델링:** 미분방정식 기반으로 상태의 시간 변화를 계산하여 주입 효과를 자연스럽게 반영.  
  - **복잡도:** ODE solver 사용, 분산 학습 지원 등으로 모델 구조와 학습 과정이 다소 복잡함.

- **코드 2 (MLP 기반):**  
  - **시계열 회귀:** 각 시점별 예측을 위해 일반적인 깊은 MLP를 사용하며, 주입 이벤트에 따른 추가 예측 손실을 활용.  
  - **데이터 확장:** 각 환자별 시계열 데이터를 확장하여 입력을 구성하는 특징.

- **코드 3 (Static-to-Sequence):**  
  - **조건부 시퀀스 생성:** 정적 입력만으로 전체 CP 시퀀스를 생성하는 방식으로, positional embedding과 Convolutional 디코더를 결합.  
  - **전체 시퀀스 예측:** 한 번의 순전파로 전체 시퀀스를 출력하는 효율적인 구조를 취함.

각 접근법은 데이터 전처리, 모델 구조 및 학습 방식에서 뚜렷한 차이를 보이며, 연구자가 어떻게 약동학(dynamic) 특성과 주입(dosing) 효과를 모델링할 것인지에 따라 선택할 수 있는 대안들을 제시합니다.