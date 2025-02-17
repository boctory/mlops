import numpy as np
import json
import requests
import tensorflow as tf
import matplotlib.pyplot as plt

# CIFAR10 클래스 이름
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_test_data():
    """테스트 데이터를 로드합니다."""
    x_test = np.load('data/cifar10/x_test.npy')
    y_test = np.load('data/cifar10/y_test.npy')
    return x_test, y_test

def predict_image(image):
    """REST API를 통해 이미지 예측을 요청합니다."""
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": image.tolist()
    })
    
    headers = {"content-type": "application/json"}
    url = "http://localhost:8501/v1/models/cifar10:predict"
    
    response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(response.text)['predictions']
    return predictions

def display_images_and_predictions(images, true_labels, predictions, num_images=5, save_path=None):
    """이미지와 예측 결과를 시각화합니다."""
    plt.figure(figsize=(15, 3))
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        pred_label = CLASS_NAMES[np.argmax(predictions[i])]
        true_label = CLASS_NAMES[true_labels[i][0]]
        plt.title(f'Pred: {pred_label}\nTrue: {true_label}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def main():
    # 테스트 데이터 로드
    print("Loading test data...")
    x_test, y_test = load_test_data()
    
    # 5개의 테스트 이미지 선택
    test_indices = np.random.choice(len(x_test), 5, replace=False)
    test_images = x_test[test_indices]
    test_labels = y_test[test_indices]
    
    # 예측 수행
    print("Making predictions...")
    predictions = predict_image(test_images)
    
    # 결과 시각화 및 저장
    print("\nDisplaying and saving results...")
    display_images_and_predictions(test_images, test_labels, predictions, save_path='prediction_results.png')

if __name__ == '__main__':
    main() 