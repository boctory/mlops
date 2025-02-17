import os
from absl import app
from pipeline import pipeline

PIPELINE_NAME = 'cifar10_training_pipeline'
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
DATA_ROOT = os.path.join('data', 'cifar10')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

def main(_):
    """Main function to run the pipeline."""
    # Create directories if they don't exist
    for directory in [PIPELINE_ROOT, DATA_ROOT, SERVING_MODEL_DIR, 
                     os.path.dirname(METADATA_PATH)]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Create and run the pipeline
    tfx_pipeline = pipeline.create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        serving_model_dir=SERVING_MODEL_DIR,
        metadata_path=METADATA_PATH
    )
    
    # Run the pipeline
    from tfx.orchestration.local.local_dag_runner import LocalDagRunner
    LocalDagRunner().run(tfx_pipeline)

if __name__ == '__main__':
    app.run(main) 