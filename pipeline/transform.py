import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""
    
    # Normalize the pixel values
    images = inputs['image']
    images = tf.cast(images, tf.float32)
    images = images / 255.0
    
    # Convert labels to integers
    labels = inputs['label']
    labels = tf.cast(labels, tf.int64)
    
    return {
        'image': images,
        'label': labels,
    } 