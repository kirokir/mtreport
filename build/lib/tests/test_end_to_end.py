"""
End-to-end integration tests for the ML training platform.
"""

import os
import json
import tempfile
import unittest
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import time

import numpy as np
import pandas as pd
from PIL import Image

from models.architectures import create_model
from data_modules.skeleton_loader import SkeletonDataLoader, validate_skeleton_schema
from data_modules.image_loader import ImageDataLoader
from data_modules.csv_loader import CSVDataLoader

class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Create necessary directories
        os.makedirs(os.path.join(self.temp_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'reports'), exist_ok=True)
        
        # Change to temp directory for tests
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_skeleton_data(self):
        """Create test skeleton data files."""
        sequences = []
        
        for seq_id in range(3):
            sequence = {
                "id": f"test_sequence_{seq_id}",
                "fps": 30,
                "frames": []
            }
            
            # Create 20 frames per sequence
            for frame_id in range(20):
                frame = {
                    "t": frame_id * 0.033,
                    "joints": {
                        "wrist": [
                            0.5 + 0.1 * np.sin(frame_id * 0.1 + seq_id),
                            0.5 + 0.1 * np.cos(frame_id * 0.1 + seq_id),
                            0.0
                        ],
                        "index": [
                            0.6 + 0.05 * np.sin(frame_id * 0.15 + seq_id),
                            0.6 + 0.05 * np.cos(frame_id * 0.15 + seq_id),
                            0.1
                        ],
                        "thumb": [
                            0.4 + 0.05 * np.sin(frame_id * 0.2 + seq_id),
                            0.4 + 0.05 * np.cos(frame_id * 0.2 + seq_id),
                            0.05
                        ]
                    },
                    "label": f"gesture_{seq_id % 2}"  # 2 classes
                }
                sequence["frames"].append(frame)
            
            sequences.append(sequence)
        
        return sequences
    
    def create_test_image_data(self):
        """Create test image data."""
        images_dir = os.path.join(self.temp_dir, 'test_images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Create test images
        for class_id in range(2):
            class_dir = os.path.join(images_dir, f'class_{class_id}')
            os.makedirs(class_dir, exist_ok=True)
            
            for img_id in range(5):
                # Create a simple colored image
                color = (255 * class_id, 100, 255 * (1 - class_id))
                image = Image.new('RGB', (64, 64), color)
                
                # Add some noise
                pixels = np.array(image)
                noise = np.random.randint(-20, 20, pixels.shape)
                pixels = np.clip(pixels + noise, 0, 255)
                image = Image.fromarray(pixels.astype(np.uint8))
                
                image_path = os.path.join(class_dir, f'image_{img_id}.png')
                image.save(image_path)
        
        return images_dir
    
    def create_test_csv_data(self):
        """Create test CSV data."""
        # Generate synthetic tabular data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Create features with some correlation to target
        X = np.random.randn(n_samples, n_features)
        
        # Create binary target based on features
        weights = np.random.randn(n_features)
        y_prob = 1 / (1 + np.exp(-(X @ weights)))
        y = (y_prob > 0.5).astype(int)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Save to CSV
        csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_skeleton_data_pipeline(self):
        """Test complete skeleton data pipeline."""
        # Create test data
        sequences = self.create_test_skeleton_data()
        
        # Save sequences to files
        skeleton_files = []
        for i, sequence in enumerate(sequences):
            file_path = os.path.join(self.temp_dir, f'skeleton_{i}.json')
            with open(file_path, 'w') as f:
                json.dump(sequence, f)
            skeleton_files.append(file_path)
        
        # Test data loading
        loader = SkeletonDataLoader()
        
        # Mock uploaded files
        mock_files = []
        for file_path in skeleton_files:
            with open(file_path, 'r') as f:
                content = f.read()
            
            mock_file = Mock()
            mock_file.name = os.path.basename(file_path)
            mock_file.read.return_value = content.encode()
            mock_file.seek = Mock()
            mock_files.append(mock_file)
        
        # Load data
        dataset_info = loader.load_from_uploaded_files(mock_files)
        
        # Verify dataset info
        self.assertEqual(dataset_info['type'], 'skeleton')
        self.assertEqual(dataset_info['num_sequences'], 3)
        self.assertEqual(dataset_info['num_classes'], 2)
        
        # Test model creation
        model = create_model('1D-CNN', (64, 9), 2)  # 3 joints * 3 coords = 9 features
        
        # Verify model can handle the data shape
        test_input = np.random.random((1, 64, 9)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 2))
    
    def test_image_data_pipeline(self):
        """Test complete image data pipeline."""
        # Create test images
        images_dir = self.create_test_image_data()
        
        # Test data loading
        loader = ImageDataLoader()
        
        # Mock uploaded files (zip file simulation)
        import zipfile
        zip_path = os.path.join(self.temp_dir, 'test_images.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zipf.write(file_path, arcname)
        
        # Mock uploaded zip file
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        mock_file = Mock()
        mock_file.name = 'test_images.zip'
        mock_file.read.return_value = zip_content
        
        # Load data
        dataset_info = loader.load_from_uploaded_files([mock_file])
        
        # Verify dataset info
        self.assertEqual(dataset_info['type'], 'image')
        self.assertEqual(dataset_info['num_samples'], 10)  # 5 images per class * 2 classes
        self.assertEqual(dataset_info['num_classes'], 2)
        
        # Test model creation
        model = create_model('CNN-Small', (224, 224, 3), 2)
        
        # Verify model can handle image data
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 2))
    
    def test_csv_data_pipeline(self):
        """Test complete CSV data pipeline."""
        # Create test CSV
        csv_path = self.create_test_csv_data()
        
        # Test data loading
        loader = CSVDataLoader()
        
        # Mock uploaded file
        with open(csv_path, 'rb') as f:
            csv_content = f.read()
        
        mock_file = Mock()
        mock_file.name = 'test_data.csv'
        mock_file.read.return_value = csv_content
        
        # Mock pandas read_csv
        df = pd.read_csv(csv_path)
        
        with patch('pandas.read_csv', return_value=df):
            dataset_info = loader.load_from_uploaded_file(mock_file)
        
        # Verify dataset info
        self.assertEqual(dataset_info['type'], 'csv')
        self.assertEqual(dataset_info['num_samples'], 100)
        self.assertEqual(dataset_info['num_features'], 10)
        
        # Test model creation
        model = create_model('MLP', (10,), 2)
        
        # Verify model can handle tabular data
        test_input = np.random.random((1, 10)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 2))
    
    def test_training_configuration(self):
        """Test training configuration creation and validation."""
        # Create a sample training config
        config = {
            'run_id': 'test_run_001',
            'run_label': 'Test Training Run',
            'dataset': 'test_skeleton_dataset',
            'model': '1D-CNN',
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 0.001,
            'class_balancing': True,
            'early_stopping': True,
            'use_mlflow': False,
            'window_length': 32
        }
        
        # Save config
        config_dir = os.path.join(self.temp_dir, 'checkpoints', config['run_id'])
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, 'run_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Verify config can be loaded
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config['run_id'], 'test_run_001')
        self.assertEqual(loaded_config['model'], '1D-CNN')
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        # Create and save a model
        model = create_model('CNN-Small', (32, 32, 3), 5)
        
        model_dir = os.path.join(self.temp_dir, 'test_model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.h5')
        model.save(model_path)
        
        # Load the model
        import tensorflow as tf
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Test that loaded model works
        test_input = np.random.random((1, 32, 32, 3)).astype(np.float32)
        
        original_output = model.predict(test_input, verbose=0)
        loaded_output = loaded_model.predict(test_input, verbose=0)
        
        # Outputs should be identical
        np.testing.assert_array_almost_equal(original_output, loaded_output)
    
    def test_report_generation_components(self):
        """Test report generation components."""
        from reports.generator import ReportGenerator
        
        # Create mock run data
        run_data = {
            'run_id': 'test_run_001',
            'config': {
                'run_label': 'Test Run',
                'model': '1D-CNN',
                'dataset': 'test_data',
                'epochs': 10,
                'learning_rate': 0.001
            },
            'results': {
                'test_results': {
                    'accuracy': 0.85,
                    'loss': 0.32,
                    'f1_score': 0.83
                },
                'model_params': 125000,
                'training_time': 150.5
            },
            'history': {
                'loss': [0.8, 0.6, 0.4, 0.35, 0.32],
                'val_loss': [0.9, 0.7, 0.5, 0.4, 0.38],
                'accuracy': [0.6, 0.7, 0.8, 0.82, 0.85],
                'val_accuracy': [0.55, 0.65, 0.75, 0.78, 0.82]
            }
        }
        
        generator = ReportGenerator()
        
        # Test executive summary generation
        summary = generator._generate_executive_summary(run_data)
        self.assertIn('85.0%', summary)
        self.assertIn('1D-CNN', summary)
        
        # Test dataset analysis
        analysis = generator._generate_dataset_analysis(run_data)
        self.assertIn('Dataset Overview', analysis)
    
    def test_data_validation_schemas(self):
        """Test data validation for all supported formats."""
        # Test skeleton validation
        valid_skeleton = {
            "id": "test_seq",
            "fps": 30,
            "frames": [
                {
                    "t": 0.0,
                    "joints": {
                        "wrist": [0.1, 0.2, 0.3],
                        "index": [0.4, 0.5, 0.6]
                    },
                    "label": "test_gesture"
                }
            ]
        }
        
        result = validate_skeleton_schema(valid_skeleton)
        self.assertTrue(result['valid'])
        
        # Test invalid skeleton (missing required field)
        invalid_skeleton = valid_skeleton.copy()
        del invalid_skeleton['fps']
        
        result = validate_skeleton_schema(invalid_skeleton)
        self.assertFalse(result['valid'])
        self.assertIn('fps', result['error'])
    
    def test_system_integration_smoke_test(self):
        """Smoke test for overall system integration."""
        # This test verifies that all major components can be imported
        # and instantiated without errors
        
        try:
            # Test data loaders
            skeleton_loader = SkeletonDataLoader()
            image_loader = ImageDataLoader()
            csv_loader = CSVDataLoader()
            
            # Test model factories
            from models.architectures import (
                CNN1DFactory, BiLSTMFactory, SmallCNNFactory,
                MobileNetV2Factory, MLPFactory
            )
            
            factories = [
                CNN1DFactory(),
                BiLSTMFactory(), 
                SmallCNNFactory(),
                MLPFactory()
            ]
            
            # Test report generator
            from reports.generator import ReportGenerator
            report_gen = ReportGenerator()
            
            # Test model utils
            from models.utils import create_callbacks, compute_model_size
            
            # If we get here, all imports and instantiations succeeded
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"System integration smoke test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling in various components."""
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_model('InvalidModelType', (10,), 2)
        
        # Test invalid skeleton schema
        invalid_data = {"invalid": "data"}
        result = validate_skeleton_schema(invalid_data)
        self.assertFalse(result['valid'])
        
        # Test file not found scenarios
        from models.utils import load_model_config
        
        with self.assertRaises(FileNotFoundError):
            load_model_config('/nonexistent/path')

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def test_model_inference_speed(self):
        """Test that models meet minimum inference speed requirements."""
        # Test small models for edge deployment
        small_models = [
            ('CNN-Small', (32, 32, 3), 5),
            ('MLP', (10,), 2),
            ('1D-CNN', (32, 9), 3)
        ]
        
        for model_type, input_shape, num_classes in small_models:
            model = create_model(model_type, input_shape, num_classes)
            
            # Benchmark inference time
            test_input = np.random.random((1,) + input_shape).astype(np.float32)
            
            # Warm up
            for _ in range(5):
                _ = model.predict(test_input, verbose=0)
            
            # Time inference
            start_time = time.time()
            for _ in range(10):
                _ = model.predict(test_input, verbose=0)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            
            # Should be faster than 100ms per inference for small models
            self.assertLess(avg_time, 0.1, 
                          f"{model_type} inference too slow: {avg_time:.3f}s")
    
    def test_memory_usage(self):
        """Test memory usage of models."""
        # Test that models don't use excessive memory
        model = create_model('CNN-Small', (64, 64, 3), 10)
        
        # Check parameter count is reasonable
        param_count = model.count_params()
        self.assertLess(param_count, 1000000, "Model has too many parameters")
        
        # Test batch processing doesn't crash
        large_batch = np.random.random((32, 64, 64, 3)).astype(np.float32)
        output = model.predict(large_batch, verbose=0)
        self.assertEqual(output.shape, (32, 10))

if __name__ == '__main__':
    # Run tests with reduced verbosity for cleaner output
    unittest.main(verbosity=1)
