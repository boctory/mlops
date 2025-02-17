import tensorflow as tf
from keras_tuner import HyperModel

class CIFAR10Model(HyperModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = tf.keras.Sequential()
        
        # First Conv2D block
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int('conv1_filters', 16, 64, step=16),
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(tf.keras.layers.AveragePooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        
        # Second Conv2D block
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int('conv2_filters', 32, 128, step=32),
            kernel_size=(3, 3),
            activation='relu'
        ))
        model.add(tf.keras.layers.AveragePooling2D((2, 2)))
        model.add(tf.keras.layers.BatchNormalization())
        
        # Dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=hp.Int('dense_units', 128, 512, step=128),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dropout(
            hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
        ))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def get_baseline_model():
    """Returns the baseline model with fixed parameters"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 