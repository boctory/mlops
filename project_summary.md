# CIFAR10 이미지 분류 모델 서빙 프로젝트

## 1. 프로젝트 개요
CIFAR10 데이터셋을 사용하여 이미지 분류 모델을 학습하고, TensorFlow Serving을 통해 모델을 서빙하는 프로젝트입니다.

## 2. 프로젝트 구조
```
mlops/
├── data/
│   └── cifar10/          # 전처리된 CIFAR10 데이터
├── models/
│   └── cifar10_model.py  # 모델 아키텍처 정의
├── serving_model/
│   └── cifar10_training_pipeline/  # 저장된 모델
├── data_preparation.py    # 데이터 전처리 스크립트
├── main.py               # 모델 학습 스크립트
├── test_model_serving.py # 모델 서빙 테스트 스크립트
├── Dockerfile            # 모델 서빙용 Dockerfile
└── requirements.txt      # 프로젝트 의존성
```

## 3. 구현 과정

### 3.1 데이터 준비
- CIFAR10 데이터셋 다운로드 및 전처리
- 이미지 정규화 (0-1 범위로 스케일링)
- NumPy 배열로 저장

```python
# data_preparation.py
def download_and_prepare_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    ...
```

### 3.2 모델 구현
- CNN 기반의 이미지 분류 모델 구현
- 배치 정규화와 드롭아웃을 통한 정규화 적용

```python
# models/cifar10_model.py
def get_baseline_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        ...
    ])
```

### 3.3 모델 학습
- 10 에포크 동안 학습 수행
- 검증 데이터로 성능 모니터링
- 최종 테스트 정확도: 67.90%

### 3.4 모델 서빙 설정
- TensorFlow Serving을 사용한 Docker 컨테이너 구성

```dockerfile
FROM tensorflow/serving:latest
COPY serving_model/cifar10_training_pipeline /models/cifar10/1
ENV MODEL_NAME=cifar10
ENV MODEL_BASE_PATH=/models/cifar10
EXPOSE 8501
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=cifar10", "--model_base_path=/models/cifar10"]
```

### 3.5 모델 서빙 테스트
- REST API를 통한 예측 요청
- 테스트 이미지에 대한 예측 결과 시각화

## 4. Troubleshooting 기록

### 4.1 패키지 설치 문제
1. TFX 버전 호환성 문제
   - 문제: `tfx>=1.12.0` 설치 실패
   - 해결: TFX 설치를 제외하고 기본 TensorFlow 패키지만 사용

2. TensorFlow와 TensorFlow Serving API 버전 충돌
   - 문제: protobuf 버전 충돌
   - 해결: TensorFlow 2.11.0과 호환되는 버전으로 통일

### 4.2 Docker 관련 문제
1. 환경 변수 문제
   - 문제: `${MODEL_NAME}`, `${MODEL_BASE_PATH}` 변수가 제대로 확장되지 않음
   - 해결: CMD 인자에 직접 값을 지정

2. 포트 충돌
   - 문제: "port is already allocated" 에러
   - 해결: 기존 컨테이너 중지 및 삭제 후 재시작

### 4.3 모델 서빙 문제
1. 모델 로드 실패
   - 문제: 모델 경로를 찾지 못하는 문제
   - 해결: Dockerfile의 COPY 명령어와 모델 경로 수정

## 5. 실행 결과

### 5.1 모델 서빙 상태 확인
```json
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```

### 5.2 예측 결과
테스트 이미지에 대한 예측 결과는 별도의 이미지 파일로 저장되어 있습니다.
- `prediction_results.png`

## 6. 분석 결과

### 6.1 모델 성능 분석

#### 전체 성능 지표
- 전체 정확도 (Accuracy): 68%
- 평균 추론 시간: 0.45ms/이미지

#### 클래스별 성능 분석
| 클래스      | Precision | Recall | F1-Score |
|------------|-----------|---------|----------|
| airplane   | 0.62      | 0.83    | 0.71     |
| automobile | 0.84      | 0.82    | 0.83     |
| bird       | 0.51      | 0.68    | 0.58     |
| cat        | 0.44      | 0.59    | 0.51     |
| deer       | 0.79      | 0.45    | 0.57     |
| dog        | 0.58      | 0.59    | 0.58     |
| frog       | 0.92      | 0.57    | 0.71     |
| horse      | 0.82      | 0.68    | 0.74     |
| ship       | 0.79      | 0.81    | 0.80     |
| truck      | 0.82      | 0.77    | 0.79     |

#### 주요 관찰사항
1. 클래스 불균형
   - 'cat'과 'bird' 클래스의 성능이 상대적으로 낮음 (F1-Score < 0.60)
   - 'automobile'과 'ship' 클래스는 높은 성능을 보임 (F1-Score > 0.80)

2. 정밀도와 재현율
   - 'frog' 클래스는 높은 정밀도(0.92)를 보이나 상대적으로 낮은 재현율(0.57)
   - 'airplane'은 반대로 낮은 정밀도(0.62)와 높은 재현율(0.83)을 보임

### 6.2 서빙 시스템 분석

#### 성능 지표
- 평균 응답 시간: 0.45ms/이미지
- 배치 처리 능력: 100개 이미지/배치
- 안정성: 에러 없이 10,000개 이미지 처리 완료

#### 시스템 특성
- REST API를 통한 안정적인 서빙
- 효율적인 배치 처리 구현
- TensorFlow Serving의 최적화된 성능 활용

### 6.3 개선 필요 사항

#### 모델 성능 개선
1. 데이터 품질
   - 'cat'과 'bird' 클래스의 학습 데이터 품질 검토
   - 클래스별 데이터 증강 전략 수립

2. 모델 아키텍처
   - 특정 클래스의 특징을 더 잘 포착할 수 있는 모델 구조 검토
   - 앙상블 기법 도입 고려

#### 서빙 시스템 개선
1. 모니터링
   - 상세한 성능 메트릭 수집 시스템 구축
   - 실시간 성능 모니터링 대시보드 구현

2. 확장성
   - 부하 분산 전략 수립
   - 자동 스케일링 구현 검토

## 7. 향후 개선 사항
1. 모델 성능 개선
   - 데이터 증강 기법 적용
   - 더 깊은 네트워크 아키텍처 시도

2. 서빙 시스템 개선
   - 배치 예측 기능 추가
   - 에러 처리 강화
   - 모니터링 시스템 구축

3. CI/CD 파이프라인 구축
   - 자동 테스트 추가
   - 모델 버전 관리 시스템 구축 