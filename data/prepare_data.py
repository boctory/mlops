import tensorflow as tf
import numpy as np
import pandas as pd
import os

def download_and_prepare_cifar10():
    """Download CIFAR10 dataset and prepare it for training."""
    # Download CIFAR10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Create directories if they don't exist
    os.makedirs('data/cifar10', exist_ok=True)
    
    # Convert images to float32 and normalize
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Create training data CSV
    train_data = pd.DataFrame({
        'image': [img.tobytes() for img in x_train],
        'label': y_train.flatten()
    })
    train_data.to_csv('data/cifar10/train.csv', index=False)
    
    # Create test data CSV
    test_data = pd.DataFrame({
        'image': [img.tobytes() for img in x_test],
        'label': y_test.flatten()
    })
    test_data.to_csv('data/cifar10/test.csv', index=False)
    
    print("Data preparation completed!")

if __name__ == '__main__':
    download_and_prepare_cifar10() 