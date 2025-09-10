"""
Convert models to ONNX format for cross-platform deployment.
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

def export_to_onnx(model: tf.keras.Model,
                  output_path: str,
                  opset_version: int = 11) -> None:
    """
    Export Keras model to ONNX format.
    
    Args:
        model: Keras model to export
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    try:
        import tf2onnx
        
        logger.info(f"Converting model to ONNX with opset version {opset_version}")
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            opset=opset_version,
            output_path=output_path
        )
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        
        logger.info(f"ONNX model saved to {output_path}")
        logger.info(f"Model size: {model_size:.2f} MB")
        
        # Test the converted model
        test_onnx_model(output_path, model.input_shape[1:])
        
    except ImportError:
        logger.error("tf2onnx not available. Install with: pip install tf2onnx")
        raise
    except Exception as e:
        logger.error(f"Failed to convert model: {e}")
        raise

def test_onnx_model(onnx_path: str, input_shape: tuple) -> None:
    """
    Test the converted ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape for testing
    """
    logger.info("Testing ONNX model...")
    
    try:
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(onnx_path)
        
        # Get input and output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        logger.info(f"Input name: {input_info.name}")
        logger.info(f"Input shape: {input_info.shape}")
        logger.info(f"Output name: {output_info.name}")
        logger.info(f"Output shape: {output_info.shape}")
        
        # Create test input
        test_input = np.random.random((1,) + input_shape).astype(np.float32)
        
        # Run inference
        output = session.run(
            [output_info.name],
            {input_info.name: test_input}
        )
        
        logger.info("ONNX model test successful!")
        logger.info(f"Output shape: {output[0].shape}")
        
    except ImportError:
        logger.error("ONNXRuntime not available. Install with: pip install onnxruntime")
        raise
    except Exception as e:
        logger.error(f"ONNX model test failed: {e}")
        raise

def benchmark_onnx_model(onnx_path: str, 
                        input_shape: tuple,
                        num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark ONNX model inference time.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape for benchmarking
        num_runs: Number of inference runs
        
    Returns:
        Dictionary of timing metrics
    """
    import time
    
    try:
        import onnxruntime as ort
        
        logger.info(f"Benchmarking ONNX model with {num_runs} runs...")
        
        # Create session
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Create test input
        test_input = np.random.random((1,) + input_shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            _ = session.run([output_name], {input_name: test_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run([output_name], {input_name: test_input})
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
    
    except ImportError:
        logger.error("ONNXRuntime not available for benchmarking")
        return {}

def optimize_onnx_model(onnx_path: str, optimized_path: str) -> None:
    """
    Optimize ONNX model for better performance.
    
    Args:
        onnx_path: Path to original ONNX model
        optimized_path: Path to save optimized model
    """
    try:
        import onnxruntime as ort
        from onnxruntime.tools import optimizer
        
        logger.info("Optimizing ONNX model...")
        
        # Load and optimize
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # or appropriate model type
            num_heads=0,  # auto-detect
            hidden_size=0  # auto-detect
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(optimized_path)
        
        # Compare sizes
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        
        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Optimized size: {optimized_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - optimized_size/original_size)*100:.1f}%")
        
    except ImportError:
        logger.warning("ONNX optimization tools not available")
    except Exception as e:
        logger.warning(f"ONNX optimization failed: {e}")

def export_model_metadata(model_dir: str, onnx_path: str, output_dir: str) -> None:
    """
    Export model metadata and conversion info.
    
    Args:
        model_dir: Source model directory
        onnx_path: Path to ONNX model
        output_dir: Output directory for metadata
    """
    # Load original config
    config_path = os.path.join(model_dir, "run_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            original_config = json.load(f)
    else:
        original_config = {}
    
    # Get ONNX model info
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path)
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        # Get file size
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        
        # Create metadata
        metadata = {
            'conversion_info': {
                'source_model': model_dir,
                'onnx_path': onnx_path,
                'conversion_date': tf.timestamp().numpy().decode(),
                'tensorflow_version': tf.__version__
            },
            'model_info': {
                'input_name': input_info.name,
                'input_shape': input_info.shape,
                'input_type': str(input_info.type),
                'output_name': output_info.name,
                'output_shape': output_info.shape,
                'output_type': str(output_info.type),
                'model_size_mb': float(onnx_size)
            },
            'original_config': original_config
        }
        
        # Save metadata
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "onnx_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
    except ImportError:
        logger.warning("ONNXRuntime not available for metadata extraction")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Export Keras models to ONNX")
    parser.add_argument("--model_dir", required=True, help="Directory containing model files")
    parser.add_argument("--output_path", required=True, help="Output path for ONNX model")
    parser.add_argument("--opset_version", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--optimize", action="store_true", help="Optimize the ONNX model")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the converted model")
    parser.add_argument("--metadata_dir", help="Directory to save metadata")
    
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_dir}")
        model, config = load_model_and_config(args.model_dir)
        
        # Export to ONNX
        export_to_onnx(
            model=model,
            output_path=args.output_path,
            opset_version=args.opset_version
        )
        
        # Optimize if requested
        if args.optimize:
            optimized_path = args.output_path.replace('.onnx', '_optimized.onnx')
            optimize_onnx_model(args.output_path, optimized_path)
        
        # Benchmark if requested
        if args.benchmark:
            input_shape = model.input_shape[1:]
            results = benchmark_onnx_model(args.output_path, input_shape)
            
            if results:
                print("\nBenchmark Results:")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.2f}")
        
        # Export metadata if requested
        if args.metadata_dir:
            export_model_metadata(args.model_dir, args.output_path, args.metadata_dir)
        
        logger.info("ONNX export completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
