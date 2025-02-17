import numpy as np
import tensorflow as tf
from pathlib import Path

def download_and_preprocess_cifar10():
    """Download and preprocess CIFAR10 dataset."""
    print("Downloading CIFAR10 dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create data directory if it doesn't exist
    data_dir = Path('data/cifar10')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(data_dir / 'x_train.npy', x_train)
    np.save(data_dir / 'y_train.npy', y_train)
    np.save(data_dir / 'x_test.npy', x_test)
    np.save(data_dir / 'y_test.npy', y_test)
    print("Dataset downloaded and preprocessed successfully!")

if __name__ == '__main__':
    download_and_preprocess_cifar10() 