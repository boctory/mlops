import tensorflow as tf
import tensorflow_transform as tft
from models.cifar10_model import CIFAR10Model
import keras_tuner as kt

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop('label')
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = tf_transform_output.transform_raw_features(
            parsed_features)
        
        outputs = model(transformed_features['image'])
        return {'outputs': outputs}
    
    return serve_tf_examples_fn

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates features and label for tuning/training."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key='label')
    
    return dataset

def run_fn(fn_args):
    """Train the model based on given args."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        batch_size=32)
    
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        batch_size=32)
    
    # Define hyperparameter tuning
    tuner = kt.Hyperband(
        CIFAR10Model(),
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='cifar10')
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    # Tune the model
    tuner.search(
        train_dataset,
        validation_data=eval_dataset,
        callbacks=[stop_early])
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    # Train the model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=fn_args.train_steps,
        callbacks=[stop_early])
    
    # Define signatures for serving
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
    }
    
    # Save the model
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures) 