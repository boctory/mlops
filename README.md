# MLOps 프로젝트: CIFAR10 이미지 분류 모델 파이프라인

## 1. 프로젝트 개요

이 프로젝트는 CIFAR10 데이터셋을 사용하여 이미지 분류 모델을 개발하고, MLOps 파이프라인을 구축하는 과정을 담고 있습니다. TensorFlow를 기반으로 하여 모델 개발부터 배포, 그리고 TFLite 변환까지의 전체 과정을 다룹니다.

### 1.1 주요 기능
- CIFAR10 데이터셋 기반 이미지 분류 모델 개발
- TensorFlow Serving을 통한 모델 서빙
- KerasTuner를 활용한 하이퍼파라미터 최적화
- TFLite 모델 변환 및 최적화

### 1.2 기술 스택
- **프레임워크**: TensorFlow 2.x
- **모델 서빙**: TensorFlow Serving
- **컨테이너화**: Docker
- **하이퍼파라미터 튜닝**: KerasTuner
- **모델 최적화**: TFLite

## 2. 프로젝트 구조

```
mlops/
├── data/
│   └── cifar10/          # CIFAR10 데이터셋
├── models/
│   └── cifar10_model.py  # 모델 아키텍처 정의
├── serving_model/
│   └── cifar10_training_pipeline/  # 저장된 모델
├── tflite_model/
│   └── model.tflite      # 변환된 TFLite 모델
├── data_preparation.py    # 데이터 전처리 스크립트
├── main.py               # 모델 학습 스크립트
├── convert_to_tflite.py  # TFLite 변환 스크립트
├── test_model_serving.py # 모델 서빙 테스트
├── Dockerfile            # TF Serving 도커파일
└── requirements.txt      # 프로젝트 의존성
```

## 3. 구현 과정

### 3.1 데이터 준비
- CIFAR10 데이터셋 다운로드 및 전처리
- 이미지 정규화 (0-1 범위로 스케일링)
- 학습/검증 데이터 분할

### 3.2 모델 개발
- CNN 기반 이미지 분류 모델 구현
- BatchNormalization과 Dropout을 통한 정규화
- KerasTuner를 활용한 하이퍼파라미터 최적화

### 3.3 모델 서빙
- TensorFlow Serving 도커 컨테이너 구성
- REST API 엔드포인트 설정
- 배치 추론 지원

### 3.4 TFLite 변환
- 모델 경량화 및 최적화
- 대표 데이터셋을 활용한 양자화
- 추론 성능 검증

## 4. 주요 결과

### 4.1 모델 성능
- **정확도**: 68% (테스트 데이터셋)
- **평균 추론 시간**: 0.45ms/이미지
- **모델 크기**: 602,698 파라미터

### 4.2 클래스별 성능
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

## 5. 실행 방법

### 5.1 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 데이터 준비
python data_preparation.py
```

### 5.2 모델 학습
```bash
# 하이퍼파라미터 튜닝 및 모델 학습
python main.py
```

### 5.3 모델 서빙
```bash
# Docker 컨테이너 실행
docker run -d --name tfserving_cifar10 -p 8501:8501 -p 8500:8500 \
  -v "$(pwd)/serving_model/cifar10_training_pipeline:/models/cifar10" \
  -e MODEL_NAME=cifar10 tensorflow/serving
```

### 5.4 TFLite 변환
```bash
# TFLite 모델 변환 및 검증
python convert_to_tflite.py
```

## 6. 프로젝트 회고

### 6.1 성과
1. **MLOps 파이프라인 구축**
   - 데이터 처리부터 모델 배포까지 자동화된 파이프라인 구현
   - 도커 기반의 안정적인 서빙 환경 구축
   - TFLite를 통한 모바일/엣지 디바이스 지원

2. **모델 최적화**
   - KerasTuner를 통한 체계적인 하이퍼파라미터 튜닝
   - 양자화를 통한 모델 경량화
   - 배치 처리를 통한 추론 성능 최적화

3. **기술 스택 통합**
   - TensorFlow 생태계 활용
   - 도커 컨테이너화
   - CI/CD 파이프라인 구축

### 6.2 개선 사항
1. **모델 성능**
   - 더 깊은 네트워크 아키텍처 검토
   - 데이터 증강 기법 적용
   - 앙상블 기법 도입 고려

2. **시스템 안정성**
   - 에러 처리 강화
   - 로깅 시스템 개선
   - 모니터링 대시보드 구축

3. **확장성**
   - 분산 학습 지원
   - 자동 스케일링 구현
   - A/B 테스트 환경 구축

### 6.3 학습한 점
1. **MLOps 실무 경험**
   - 엔드투엔드 ML 파이프라인 구축 경험
   - 도커 기반 서비스 배포 실습
   - 모델 최적화 및 경량화 기법 습득

2. **기술적 인사이트**
   - TensorFlow 생태계의 이해
   - 하이퍼파라미터 튜닝 방법론
   - 모델 서빙 아키텍처 설계

3. **프로젝트 관리**
   - 체계적인 코드 구조화
   - 문서화의 중요성
   - 버전 관리 및 협업 방식

## 7. 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [TensorFlow Serving 가이드](https://www.tensorflow.org/tfx/serving/docker)
- [KerasTuner 문서](https://keras.io/keras_tuner/)
- [TFLite 변환 가이드](https://www.tensorflow.org/lite/convert)
- [박찬성님의 Semantic Segmentation ML 파이프라인](https://github.com/deep-diver/semantic-segmentation-ml-pipeline) 