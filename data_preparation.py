import tensorflow as tf
import numpy as np
import os

def download_and_prepare_cifar10():
    """Download CIFAR10 dataset and prepare it for training."""
    print("Downloading and preparing CIFAR10 dataset...")
    
    # Download CIFAR10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Create directories
    os.makedirs('data/cifar10', exist_ok=True)
    os.makedirs('serving_model', exist_ok=True)
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Save the data
    np.save('data/cifar10/x_train.npy', x_train)
    np.save('data/cifar10/y_train.npy', y_train)
    np.save('data/cifar10/x_test.npy', x_test)
    np.save('data/cifar10/y_test.npy', y_test)
    
    print("Dataset preparation completed!")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

if __name__ == '__main__':
    download_and_prepare_cifar10() 