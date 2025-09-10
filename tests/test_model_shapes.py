"""
Tests for model architectures and shape validation.
"""

import unittest
import numpy as np
import tensorflow as tf
from typing import Tuple

from models.architectures import (
    CNN1DFactory, BiLSTMFactory, HybridCNNLSTMFactory,
    SmallCNNFactory, MobileNetV2Factory, MLPFactory,
    get_model_factory, create_model
)

class TestModelShapes(unittest.TestCase):
    """Test cases for model architecture shapes and parameters."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
    
    def test_cnn1d_factory_shapes(self):
        """Test 1D CNN model shapes for skeleton data."""
        factory = CNN1DFactory()
        
        # Test with typical skeleton input shape
        input_shape = (64, 18)  # 64 timesteps, 18 features (6 joints * 3 coords)
        num_classes = 5
        
        model = factory.create_model(input_shape, num_classes)
        
        # Check input/output shapes
        self.assertEqual(model.input_shape, (None, 64, 18))
        self.assertEqual(model.output_shape, (None, 5))
        
        # Test forward pass
        test_input = np.random.random((1, 64, 18)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        self.assertEqual(output.shape, (1, 5))
        
        # Check output is probability distribution
        self.assertAlmostEqual(np.sum(output[0]), 1.0, places=5)
        self.assertTrue(np.all(output[0] >= 0))
    
    def test_bilstm_factory_shapes(self):
        """Test BiLSTM model shapes."""
        factory = BiLSTMFactory()
        
        input_shape = (32, 12)  # 32 timesteps, 12 features
        num_classes = 3
        
        model = factory.create_model(input_shape, num_classes)
        
        # Check shapes
        self.assertEqual(model.input_shape, (None, 32, 12))
        self.assertEqual(model.output_shape, (None, 3))
        
        # Test forward pass
        test_input = np.random.random((2, 32, 12)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        self.assertEqual(output.shape, (2, 3))
    
    def test_hybrid_cnn_lstm_shapes(self):
        """Test hybrid CNN-LSTM model shapes."""
        factory = HybridCNNLSTMFactory()
        
        input_shape = (100, 15)
        num_classes = 4
        
        model = factory.create_model(input_shape, num_classes)
        
        # Check shapes
        self.assertEqual(model.input_shape, (None, 100, 15))
        self.assertEqual(model.output_shape, (None, 4))
        
        # Test forward pass
        test_input = np.random.random((1, 100, 15)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        self.assertEqual(output.shape, (1, 4))
    
    def test_small_cnn_shapes(self):
        """Test small CNN model shapes for images."""
        factory = SmallCNNFactory()
        
        input_shape = (224, 224, 3)  # RGB image
        num_classes = 10
        
        model = factory.create_model(input_shape, num_classes)
        
        # Check shapes
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 10))
        
        # Test forward pass with small batch
        test_input = np.random.random((2, 224, 224, 3)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        self.assertEqual(output.shape, (2, 10))
    
    def test_mobilenetv2_shapes(self):
        """Test MobileNetV2 model shapes."""
        factory = MobileNetV2Factory()
        
        input_shape = (224, 224, 3)
        num_classes = 1000
        
        model = factory.create_model(input_shape, num_classes)
        
        # Check shapes
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 1000))
        
        # Test forward pass
        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        self.assertEqual(output.shape, (1, 1000))
    
    def test_mlp_shapes(self):
        """Test MLP model shapes for tabular data."""
        factory = MLPFactory()
        
        input_shape = (20,)  # 20 features
        num_classes = 2
        
        model = factory.create_model(input_shape, num_classes)
        
        # Check shapes
        self.assertEqual(model.input_shape, (None, 20))
        self.assertEqual(model.output_shape, (None, 2))
        
        # Test forward pass
        test_input = np.random.random((5, 20)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        
        self.assertEqual(output.shape, (5, 2))
    
    def test_model_factory_registry(self):
        """Test model factory registry functionality."""
        # Test valid model types
        valid_types = ['1D-CNN', 'BiLSTM', 'CNN-Small', 'MobileNetV2', 'MLP']
        
        for model_type in valid_types:
            factory = get_model_factory(model_type)
            self.assertIsNotNone(factory)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            get_model_factory('InvalidModel')
    
    def test_create_model_convenience_function(self):
        """Test the convenience create_model function."""
        input_shape = (50, 10)
        num_classes = 3
        
        model = create_model('1D-CNN', input_shape, num_classes)
        
        self.assertEqual(model.input_shape, (None, 50, 10))
        self.assertEqual(model.output_shape, (None, 3))
    
    def test_model_parameters_count(self):
        """Test model parameter counts are reasonable."""
        test_cases = [
            ('1D-CNN', (64, 18), 5, 50000, 2000000),     # 50K - 2M params
            ('BiLSTM', (32, 12), 3, 100000, 5000000),    # 100K - 5M params  
            ('CNN-Small', (64, 64, 3), 10, 10000, 1000000), # 10K - 1M params
            ('MLP', (20,), 2, 1000, 500000),             # 1K - 500K params
        ]
        
        for model_type, input_shape, num_classes, min_params, max_params in test_cases:
            model = create_model(model_type, input_shape, num_classes)
            param_count = model.count_params()
            
            self.assertGreaterEqual(param_count, min_params, 
                                  f"{model_type} has too few parameters: {param_count}")
            self.assertLessEqual(param_count, max_params,
                               f"{model_type} has too many parameters: {param_count}")
    
    def test_model_compilation(self):
        """Test that all models compile successfully."""
        test_configs = [
            ('1D-CNN', (64, 18), 5),
            ('BiLSTM', (32, 12), 3),
            ('CNN-Small', (32, 32, 3), 5),
            ('MLP', (15,), 2)
        ]
        
        for model_type, input_shape, num_classes in test_configs:
            model = create_model(model_type, input_shape, num_classes)
            
            # Check model is compiled
            self.assertIsNotNone(model.optimizer)
            self.assertIsNotNone(model.loss)
            self.assertIsNotNone(model.metrics)
    
    def test_custom_model_parameters(self):
        """Test models with custom parameters."""
        factory = CNN1DFactory()
        
        # Test with custom parameters
        custom_params = {
            'filters': (32, 64, 128),
            'kernel_sizes': (7, 5, 3),
            'dropout_rate': 0.5
        }
        
        model = factory.create_model(
            input_shape=(64, 18),
            num_classes=5,
            **custom_params
        )
        
        # Model should still work
        test_input = np.random.random((1, 64, 18)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 5))
    
    def test_edge_case_shapes(self):
        """Test models with edge case input shapes."""
        # Very small sequence
        model = create_model('1D-CNN', (8, 6), 2)
        test_input = np.random.random((1, 8, 6)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 2))
        
        # Single feature
        model = create_model('MLP', (1,), 2)
        test_input = np.random.random((1, 1)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 2))
        
        # Very small image
        model = create_model('CNN-Small', (32, 32, 1), 3)
        test_input = np.random.random((1, 32, 32, 1)).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        self.assertEqual(output.shape, (1, 3))

class TestModelMemoryUsage(unittest.TestCase):
    """Test memory usage of models."""
    
    def test_model_memory_efficiency(self):
        """Test that models don't use excessive memory."""
        # Test with larger batch sizes to check memory scaling
        model = create_model('CNN-Small', (64, 64, 3), 5)
        
        # Test with progressively larger batch sizes
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            test_input = np.random.random((batch_size, 64, 64, 3)).astype(np.float32)
            output = model.predict(test_input, verbose=0)
            self.assertEqual(output.shape, (batch_size, 5))
    
    def test_model_gradient_flow(self):
        """Test that gradients can flow through models."""
        model = create_model('1D-CNN', (32, 10), 3)
        
        # Create dummy data
        x = tf.random.normal((4, 32, 10))
        y = tf.one_hot([0, 1, 2, 0], depth=3)
        
        # Test gradient computation
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check that we get gradients for all trainable variables
        self.assertEqual(len(gradients), len(model.trainable_variables))
        
        # Check that gradients are not None or all zeros
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(tf.reduce_all(tf.equal(grad, 0.0)))

if __name__ == '__main__':
    unittest.main()
