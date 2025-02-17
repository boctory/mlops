import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Trainer
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.proto import trainer_pb2
from tfx.proto import pusher_pb2
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor

def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    serving_model_dir: str,
    metadata_path: str,
    beam_pipeline_args: list = None
):
    """Creates a TFX pipeline for training and deploying the CIFAR10 model."""
    
    # Create components
    example_gen = CsvExampleGen(input_base=data_root)
    
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
    
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'])
    
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath('pipeline/transform.py'))
    
    trainer = Trainer(
        module_file=os.path.abspath('pipeline/trainer.py'),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))
    
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
                'latest_blessed_model_resolver')
    
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=tfx.proto.eval_config_pb2.EvalConfig())
    
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
    
    # Define the pipeline
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path)
    ) 