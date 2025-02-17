import tensorflow as tf
import numpy as np
import os

def load_saved_model():
    """저장된 모델을 로드합니다."""
    print("Loading SavedModel...")
    model = tf.saved_model.load('serving_model/cifar10_training_pipeline')
    return model

def convert_to_tflite():
    """SavedModel을 TFLite 모델로 변환합니다."""
    print("\nConverting to TFLite format...")
    
    # TFLite 변환기 생성
    converter = tf.lite.TFLiteConverter.from_saved_model('serving_model/cifar10_training_pipeline')
    
    # 최적화 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # 대표 데이터셋 생성 (양자화를 위한 캘리브레이션)
    def representative_dataset():
        # 테스트 데이터 로드
        x_test = np.load('data/cifar10/x_test.npy')
        for i in range(100):  # 100개의 샘플만 사용
            yield [x_test[i:i+1].astype(np.float32)]
    
    # 양자화 설정
    converter.representative_dataset = representative_dataset
    
    # 변환 실행
    tflite_model = converter.convert()
    
    # 모델 저장
    os.makedirs('tflite_model', exist_ok=True)
    with open('tflite_model/model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved to: tflite_model/model.tflite")
    
    return tflite_model

def verify_tflite_model():
    """TFLite 모델을 로드하고 서명을 확인합니다."""
    print("\nVerifying TFLite model...")
    
    # 모델 로드
    interpreter = tf.lite.Interpreter(model_path='tflite_model/model.tflite')
    interpreter.allocate_tensors()
    
    # 입력/출력 세부정보 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel Signature Details:")
    print("\nInput Details:")
    for detail in input_details:
        print(f"- Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
    
    print("\nOutput Details:")
    for detail in output_details:
        print(f"- Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
    
    # 테스트 추론 실행
    print("\nRunning test inference...")
    x_test = np.load('data/cifar10/x_test.npy')
    test_image = x_test[0:1].astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nTest inference successful!")
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {np.argmax(output)}")

def main():
    """메인 실행 함수"""
    # SavedModel을 TFLite로 변환
    convert_to_tflite()
    
    # TFLite 모델 검증
    verify_tflite_model()
    
    print("\nConversion and verification completed successfully!")

if __name__ == '__main__':
    main() 