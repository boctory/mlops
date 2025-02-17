import numpy as np
import json
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# CIFAR10 클래스 이름
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_test_data():
    """테스트 데이터를 로드합니다."""
    x_test = np.load('data/cifar10/x_test.npy')
    y_test = np.load('data/cifar10/y_test.npy')
    return x_test, y_test

def predict_batch(images):
    """REST API를 통해 배치 예측을 요청합니다."""
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": images.tolist()
    })
    
    headers = {"content-type": "application/json"}
    url = "http://localhost:8501/v1/models/cifar10:predict"
    
    start_time = time.time()
    response = requests.post(url, data=data, headers=headers)
    end_time = time.time()
    
    predictions = json.loads(response.text)['predictions']
    return predictions, end_time - start_time

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """혼동 행렬을 시각화합니다."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_performance():
    """모델의 성능을 분석합니다."""
    print("Loading test data...")
    x_test, y_test = load_test_data()
    y_test = y_test.flatten()
    
    print("Making predictions...")
    batch_size = 100
    predictions = []
    total_time = 0
    
    for i in range(0, len(x_test), batch_size):
        batch_images = x_test[i:i+batch_size]
        batch_preds, inference_time = predict_batch(batch_images)
        predictions.extend(batch_preds)
        total_time += inference_time
        print(f"Processed {i+len(batch_images)}/{len(x_test)} images")
    
    predictions = np.array(predictions)
    y_pred = np.argmax(predictions, axis=1)
    
    # 성능 메트릭 계산
    print("\nPerformance Metrics:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # 평균 추론 시간
    avg_time_per_image = total_time / len(x_test)
    print(f"\nAverage inference time per image: {avg_time_per_image*1000:.2f}ms")
    
    # 혼동 행렬 시각화
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    # 클래스별 정확도
    class_accuracy = {}
    for i in range(len(CLASS_NAMES)):
        mask = (y_test == i)
        class_accuracy[CLASS_NAMES[i]] = (y_pred[mask] == i).mean()
    
    # 클래스별 정확도 시각화
    plt.figure(figsize=(10, 5))
    plt.bar(class_accuracy.keys(), class_accuracy.values())
    plt.title('Class-wise Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.close()

if __name__ == '__main__':
    analyze_performance() 