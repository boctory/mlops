import tensorflow as tf
import numpy as np
import os
from models.cifar10_model import CIFAR10Model
import keras_tuner as kt

def load_data():
    """Load preprocessed CIFAR10 dataset."""
    x_train = np.load('data/cifar10/x_train.npy')
    y_train = np.load('data/cifar10/y_train.npy')
    x_test = np.load('data/cifar10/x_test.npy')
    y_test = np.load('data/cifar10/y_test.npy')
    return (x_train, y_train), (x_test, y_test)

def train_and_evaluate():
    """Train and evaluate the model with hyperparameter tuning."""
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create the hyperparameter tuner
    print("\nInitializing hyperparameter tuning...")
    tuner = kt.Hyperband(
        CIFAR10Model(),
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='cifar10'
    )
    
    # Configure early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Perform hyperparameter search
    print("\nStarting hyperparameter search...")
    tuner.search(
        x_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        callbacks=[stop_early]
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest hyperparameters:")
    print(f"- Conv1 filters: {best_hps.get('conv1_filters')}")
    print(f"- Conv2 filters: {best_hps.get('conv2_filters')}")
    print(f"- Dense units: {best_hps.get('dense_units')}")
    print(f"- Dropout rate: {best_hps.get('dropout_rate')}")
    print(f"- Learning rate: {best_hps.get('learning_rate')}")
    
    # Build and train the model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_split=0.2,
        batch_size=64,
        callbacks=[stop_early]
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save the model
    print("\nSaving model for serving...")
    save_path = os.path.join('serving_model', 'cifar10_training_pipeline')
    tf.saved_model.save(model, save_path)
    print(f"Model saved successfully at: {save_path}")

if __name__ == '__main__':
    train_and_evaluate() 