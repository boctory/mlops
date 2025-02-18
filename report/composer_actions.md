# 컴포저 작업 내용 정리

## 1. 초기 설정 및 파일 구조화

### 1.1 프로젝트 구조 설정
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

### 1.2 의존성 관리
- requirements.txt 파일 생성 및 필요한 패키지 정의
- 가상환경 설정 및 패키지 설치

## 2. 데이터 관리

### 2.1 데이터 다운로드 스크립트
- `download_dataset.py` 생성
- CIFAR10 데이터셋 다운로드 및 전처리 기능 구현
- 데이터 정규화 및 저장 로직 구현

### 2.2 모델 다운로드 스크립트
- `download_model.py` 생성
- 클라우드 스토리지에서 모델 다운로드 템플릿 구현
- 다양한 스토리지 서비스 지원 준비

## 3. Git 관리

### 3.1 .gitignore 설정
```
# Python 관련
__pycache__/
*.py[cod]
*.so
.Python
build/
...

# 가상환경
.env
.venv
env/
venv/
ENV/

# 데이터 및 모델 파일
data/cifar10/*.npy
*.h5
*.pb
*.dylib
serving_model/
hyperparameter_tuning/
tflite_model/
```

### 3.2 Git 저장소 초기화 및 파일 업로드
1. 저장소 초기화
```bash
rm -rf .git
git init
```

2. 파일 추가 및 커밋
```bash
git add .
git commit -m "feat: Initial commit with project structure"
```

3. 원격 저장소 연결 및 푸시
```bash
git remote add origin https://github.com/boctory/mlops.git
git push -f origin main
```

## 4. 문서화

### 4.1 프로젝트 문서
- README.md 작성
- 프로젝트 구조 및 실행 방법 설명
- 기술 스택 및 주요 기능 정리

### 4.2 분석 문서
- tflite_conversion_results.md
- tfx_project_analysis.md
- hfpusher_analysis.md
- tuning_result.md

## 5. 발생한 문제 및 해결

### 5.1 대용량 파일 문제
- 문제: GitHub의 파일 크기 제한으로 인한 푸시 실패
  - data/cifar10/x_test.npy (117.19 MB)
  - data/cifar10/x_train.npy (585.94 MB)
  - .venv/lib/python3.11/site-packages/tensorflow/libtensorflow_cc.2.dylib (661.36 MB)

- 해결:
  1. .gitignore에 대용량 파일 추가
  2. 데이터 다운로드 스크립트 제공
  3. 모델 다운로드 스크립트 제공

### 5.2 가상환경 관리
- 문제: 가상환경 디렉토리의 Git 추적
- 해결: .gitignore에 가상환경 관련 경로 추가

## 6. 향후 작업 계획

### 6.1 즉시 필요한 작업
1. 데이터 다운로드 스크립트 구현 완료
2. 모델 다운로드 스크립트 구현 완료
3. CI/CD 파이프라인 구축

### 6.2 중장기 개선사항
1. Git LFS 도입 검토
2. 자동화된 테스트 추가
3. 모니터링 시스템 구축 