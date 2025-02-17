FROM tensorflow/serving:latest

# Copy the SavedModel to the container
COPY serving_model/cifar10_training_pipeline /models/cifar10/1

# Set environment variables
ENV MODEL_NAME=cifar10
ENV MODEL_BASE_PATH=/models/cifar10

# Expose the port
EXPOSE 8501

# Start TensorFlow Serving
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=cifar10", "--model_base_path=/models/cifar10"] 