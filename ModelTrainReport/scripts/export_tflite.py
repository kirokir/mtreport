"""
Export models to TensorFlow Lite format for edge deployment.
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_config(model_dir: str) -> tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Load model and configuration from checkpoint directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Tuple of (model, config)
    """
    # Load model
    model_path = os.path.join(model_dir, "model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    # Load configuration
    config_path = os.path.join(model_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    return model, config

def export_to_tflite(model: tf.keras.Model,
                    output_path: str,
                    quantization: str = "float16",
                    representative_dataset: Optional[np.ndarray] = None) -> None:
    """
    Export Keras model to TensorFlow Lite format.
    
    Args:
        model: Keras model to export
        output_path: Path to save TFLite model
        quantization: Quantization type ("float16", "int8", "dynamic")
        representative_dataset: Representative dataset for int8 quantization
    """
    logger.info(f"Converting model to TensorFlow Lite with {quantization} quantization")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure optimization
    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None:
            def representative_data_gen():
                for sample in representative_dataset[:100]:  # Use first 100 samples
                    yield [sample.reshape(1, *sample.shape).astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            logger.warning("No representative dataset provided for int8 quantization")
    
    elif quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert model
    try:
        tflite_model = converter.convert()
        
        # Save TFLite model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = len(tflite_model) / (1024 * 1024)  # Size in MB
        
        logger.info(f"TFLite model saved to {output_path}")
        logger.info(f"Model size: {model_size:.2f} MB")
        
        # Test the converted model
        test_tflite_model(output_path, model.input_shape[1:])
        
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        raise

def test_tflite_model(tflite_path: str, input_shape: tuple) -> None:
    """
    Test the converted TFLite model.
    
    Args:
        tflite_path: Path to TFLite model
        input_shape: Input shape for testing
    """
    logger.info("Testing TFLite model...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Input shape: {input_details[0]['shape']}")
        logger.info(f"Output shape: {output_details[0]['shape']}")
        
        # Create test input
        test_input = np.random.random((1,) + input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info("TFLite model test successful!")
        logger.info(f"Output shape: {output.shape}")
        
    except Exception as e:
        logger.error(f"TFLite model test failed: {e}")
        raise

def benchmark_tflite_model(tflite_path: str, 
                          input_shape: tuple,
                          num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark TFLite model inference time.
    
    Args:
        tflite_path: Path to TFLite model
        input_shape: Input shape for benchmarking
        num_runs: Number of inference runs
        
    Returns:
        Dictionary of timing metrics
    """
    import time
    
    logger.info(f"Benchmarking TFLite model with {num_runs} runs...")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create test input
    test_input = np.random.random((1,) + input_shape).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    results = {
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'min_time_ms': float(np.min(times) * 1000),
        'max_time_ms': float(np.max(times) * 1000),
        'median_time_ms': float(np.median(times) * 1000)
    }
    
    logger.info(f"Average inference time: {results['mean_time_ms']:.2f} ms")
    
    return results

def export_model_metadata(model_dir: str, tflite_path: str, output_dir: str) -> None:
    """
    Export model metadata and conversion info.
    
    Args:
        model_dir: Source model directory
        tflite_path: Path to TFLite model
        output_dir: Output directory for metadata
    """
    # Load original config
    config_path = os.path.join(model_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            original_config = json.load(f)
    else:
        original_config = {}
    
    # Get TFLite model info
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get file size
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    
    # Create metadata
    metadata = {
        'conversion_info': {
            'source_model': model_dir,
            'tflite_path': tflite_path,
            'conversion_date': tf.timestamp().numpy().decode(),
            'tensorflow_version': tf.__version__
        },
        'model_info': {
            'input_shape': input_details[0]['shape'].tolist(),
            'input_dtype': str(input_details[0]['dtype']),
            'output_shape': output_details[0]['shape'].tolist(),
            'output_dtype': str(output_details[0]['dtype']),
            'model_size_mb': float(tflite_size)
        },
        'original_config': original_config
    }
    
    # Save metadata
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "tflite_metadata.json")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Export Keras models to TensorFlow Lite")
    parser.add_argument("--model_dir", required=True, help="Directory containing model files")
    parser.add_argument("--output_path", required=True, help="Output path for TFLite model")
    parser.add_argument("--quantization", choices=["float16", "int8", "dynamic"], 
                       default="float16", help="Quantization type")
    parser.add_argument("--representative_data", help="Path to representative dataset (for int8)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the converted model")
    parser.add_argument("--metadata_dir", help="Directory to save metadata")
    
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_dir}")
        model, config = load_model_and_config(args.model_dir)
        
        # Load representative data if provided
        representative_data = None
        if args.representative_data and os.path.exists(args.representative_data):
            logger.info(f"Loading representative data from {args.representative_data}")
            representative_data = np.load(args.representative_data)
        
        # Export to TFLite
        export_to_tflite(
            model=model,
            output_path=args.output_path,
            quantization=args.quantization,
            representative_dataset=representative_data
        )
        
        # Benchmark if requested
        if args.benchmark:
            input_shape = model.input_shape[1:]
            results = benchmark_tflite_model(args.output_path, input_shape)
            
            print("\nBenchmark Results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.2f}")
        
        # Export metadata if requested
        if args.metadata_dir:
            export_model_metadata(args.model_dir, args.output_path, args.metadata_dir)
        
        logger.info("TFLite export completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
