# Semantic Segmentation ML 파이프라인 프로젝트 분석

## 1. 프로젝트 개요

### 1.1 프로젝트 목적
- TensorFlow Extended(TFX)를 활용한 Semantic Segmentation 모델의 엔드투엔드 ML 파이프라인 구축
- GCP의 Vertex AI 서비스들과 Hugging Face 통합
- 실제 프로덕션 환경에서 활용 가능한 ML 파이프라인 아키텍처 제시

### 1.2 주요 기술 스택
- **ML 프레임워크**: TensorFlow, TFX
- **클라우드 서비스**: Google Cloud Platform (Vertex AI, Vertex Pipeline, Vertex Training, Vertex Endpoint)
- **모델 저장소**: Hugging Face Hub
- **데모 애플리케이션**: Gradio
- **데이터 포맷**: TFRecord

## 2. 프로젝트 구조 분석

### 2.1 디렉토리 구조
```
project
├── notebooks/
│   ├── gradio_demo.ipynb
│   ├── inference_from_SavedModel.ipynb
│   ├── parse_tfrecords_pets.ipynb
│   └── tfx_pipeline.ipynb
├── tfrecords/
│   └── create_tfrecords_pets.py
└── training_pipeline/
    ├── apps/          # Gradio 앱 템플릿
    ├── models/        # 모델 관련 파일
    └── pipeline/      # TFX 파이프라인 정의
```

### 2.2 핵심 컴포넌트
1. **데이터 처리**
   - TFRecord 형식 활용
   - ExampleGen, SchemaGen 컴포넌트 구현
   - ImportSchemaGen으로 향상된 TFRecord 파싱 기능

2. **모델 학습**
   - U-NET 아키텍처 기반 구현
   - Trainer, Evaluator 컴포넌트 통합
   - Vertex Training 활용

3. **모델 배포**
   - Custom HFPusher 컴포넌트 개발
   - Hugging Face Hub 자동 배포
   - Gradio 데모 앱 자동 생성

## 3. 파이프라인 특징

### 3.1 실행 환경 지원
1. **로컬 환경**
```bash
tfx pipeline create --pipeline-path=local_runner.py --engine=local
tfx pipeline compile --pipeline-path=local_runner.py --engine=local
tfx run create --pipeline-name=segformer-training-pipeline --engine=local
```

2. **Vertex AI 환경**
```bash
tfx pipeline create --pipeline-path=kubeflow_runner.py --engine=vertex
tfx pipeline compile --pipeline-path=kubeflow_runner.py --engine=vertex
tfx run create --pipeline-name=segformer-training-pipeline --engine=vertex
```

### 3.2 CI/CD 통합
- GitHub Actions를 통한 자동화
- workflow_dispatch 기능으로 파이프라인 실행 제어
- GCP 프로젝트 ID 자동 주입 기능

## 4. 주요 기술적 특징

### 4.1 커스텀 컴포넌트
1. **HFPusher**
   - Hugging Face Hub 연동
   - 모델 자동 업로드
   - Gradio 앱 배포 자동화

### 4.2 데이터 처리 최적화
1. **TFRecord 활용**
   - 효율적인 데이터 처리
   - 스키마 기반 검증
   - 대규모 데이터셋 지원

### 4.3 확장성 고려사항
1. **Dataflow 통합**
   - 대규모 데이터셋 처리 지원
   - 분산 처리 가능성
   - 성능 최적화

## 5. 프로젝트 특이사항

### 5.1 데이터셋 전환
- 초기: Sidewalks 데이터셋
- 최종: PETS 데이터셋
- 전환 이유: 다운샘플링 영향 최소화

### 5.2 모델 선택
- U-NET 기반 아키텍처 채택
- 빠른 실험 및 검증 가능
- 향후 SegFormer, DeepLabv3+ 확장 가능성

## 6. 향후 개선 방향

### 6.1 모델 개선
- SOTA 모델 통합 (SegFormer, DeepLabv3+)
- 고해상도 이미지 처리 최적화
- 모델 성능 향상

### 6.2 파이프라인 개선
- Dataflow 완전 통합
- 모니터링 시스템 강화
- 자동화 범위 확대

## 7. 결론

이 프로젝트는 TFX를 활용한 실제 프로덕션 수준의 ML 파이프라인 구축 방법을 보여주는 훌륭한 예시입니다. 특히 다음 측면에서 가치가 있습니다:

1. **엔드투엔드 자동화**
   - 데이터 처리부터 모델 배포까지 완전 자동화
   - CI/CD 통합으로 지속적 배포 가능

2. **확장성**
   - 클라우드 네이티브 설계
   - 대규모 데이터셋 처리 고려
   - 다양한 모델 아키텍처 지원 가능

3. **실용성**
   - 실제 프로덕션 환경 고려
   - 다양한 실행 환경 지원
   - 모니터링 및 관리 용이성

이 프로젝트는 ML 파이프라인 구축을 위한 실질적인 참고자료로서 큰 가치가 있으며, MLOps 실무에 직접 적용 가능한 많은 인사이트를 제공합니다. 