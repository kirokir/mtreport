"""
Model Architectures
Factory classes for creating different model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

logger = logging.getLogger(__name__)

class ModelFactory(ABC):
    """Abstract base class for model factories."""
    
    @abstractmethod
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int, 
                    **kwargs) -> tf.keras.Model:
        """Create and return a compiled model."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this model type."""
        pass

class CNN1DFactory(ModelFactory):
    """Factory for 1D CNN models (skeleton data)."""
    
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    filters: Tuple[int, ...] = (64, 128, 256),
                    kernel_sizes: Tuple[int, ...] = (5, 3, 3),
                    pool_sizes: Tuple[int, ...] = (2, 2, 2),
                    dropout_rate: float = 0.3,
                    **kwargs) -> tf.keras.Model:
        """
        Create 1D CNN model for skeleton data.
        
        Args:
            input_shape: Input shape (T, F) where T=time steps, F=features
            num_classes: Number of output classes
            filters: Number of filters for each conv layer
            kernel_sizes: Kernel sizes for each conv layer
            pool_sizes: Pool sizes for each pooling layer
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First conv block
            layers.Conv1D(filters[0], kernel_sizes[0], padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=pool_sizes[0]),
            layers.Dropout(dropout_rate),
            
            # Second conv block
            layers.Conv1D(filters[1], kernel_sizes[1], padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=pool_sizes[1]),
            layers.Dropout(dropout_rate),
            
            # Third conv block
            layers.Conv1D(filters[2], kernel_sizes[2], padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"Created 1D-CNN model with {model.count_params()} parameters")
        return model
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for 1D CNN."""
        return {
            'filters': (64, 128, 256),
            'kernel_sizes': (5, 3, 3),
            'pool_sizes': (2, 2, 2),
            'dropout_rate': 0.3
        }

class BiLSTMFactory(ModelFactory):
    """Factory for Bidirectional LSTM models."""
    
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    lstm_units: Tuple[int, ...] = (128, 64),
                    dropout_rate: float = 0.3,
                    recurrent_dropout: float = 0.2,
                    **kwargs) -> tf.keras.Model:
        """
        Create Bidirectional LSTM model.
        
        Args:
            input_shape: Input shape (T, F)
            num_classes: Number of output classes
            lstm_units: Units for each LSTM layer
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First BiLSTM layer
            layers.Bidirectional(
                layers.LSTM(lstm_units[0], 
                          return_sequences=True,
                          dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout)
            ),
            layers.BatchNormalization(),
            
            # Second BiLSTM layer
            layers.Bidirectional(
                layers.LSTM(lstm_units[1],
                          return_sequences=False,
                          dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout)
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"Created BiLSTM model with {model.count_params()} parameters")
        return model
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for BiLSTM."""
        return {
            'lstm_units': (128, 64),
            'dropout_rate': 0.3,
            'recurrent_dropout': 0.2
        }

class HybridCNNLSTMFactory(ModelFactory):
    """Factory for hybrid CNN-LSTM models."""
    
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    cnn_filters: Tuple[int, ...] = (64, 128),
                    cnn_kernels: Tuple[int, ...] = (5, 3),
                    lstm_units: int = 128,
                    dropout_rate: float = 0.3,
                    **kwargs) -> tf.keras.Model:
        """
        Create hybrid CNN-LSTM model.
        
        Args:
            input_shape: Input shape (T, F)
            num_classes: Number of output classes
            cnn_filters: Filters for CNN layers
            cnn_kernels: Kernel sizes for CNN layers
            lstm_units: LSTM units
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # CNN feature extraction
            layers.Conv1D(cnn_filters[0], cnn_kernels[0], padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(cnn_filters[1], cnn_kernels[1], padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # LSTM temporal modeling
            layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False)),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"Created CNN-LSTM hybrid model with {model.count_params()} parameters")
        return model
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for CNN-LSTM hybrid."""
        return {
            'cnn_filters': (64, 128),
            'cnn_kernels': (5, 3),
            'lstm_units': 128,
            'dropout_rate': 0.3
        }

class SmallCNNFactory(ModelFactory):
    """Factory for small CNN models (image data)."""
    
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    filters: Tuple[int, ...] = (32, 64, 128),
                    kernel_size: int = 3,
                    dropout_rate: float = 0.25,
                    **kwargs) -> tf.keras.Model:
        """
        Create small CNN model for image classification.
        
        Args:
            input_shape: Input shape (H, W, C)
            num_classes: Number of output classes
            filters: Number of filters for each conv layer
            kernel_size: Kernel size for conv layers
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First conv block
            layers.Conv2D(filters[0], kernel_size, activation='relu'),
            layers.MaxPooling2D(2),
            layers.BatchNormalization(),
            
            # Second conv block
            layers.Conv2D(filters[1], kernel_size, activation='relu'),
            layers.MaxPooling2D(2),
            layers.BatchNormalization(),
            
            # Third conv block
            layers.Conv2D(filters[2], kernel_size, activation='relu'),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"Created Small CNN model with {model.count_params()} parameters")
        return model
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for Small CNN."""
        return {
            'filters': (32, 64, 128),
            'kernel_size': 3,
            'dropout_rate': 0.25
        }

class MobileNetV2Factory(ModelFactory):
    """Factory for MobileNetV2 transfer learning models."""
    
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    trainable_layers: int = 20,
                    dropout_rate: float = 0.2,
                    **kwargs) -> tf.keras.Model:
        """
        Create MobileNetV2 transfer learning model.
        
        Args:
            input_shape: Input shape (H, W, C)
            num_classes: Number of output classes
            trainable_layers: Number of top layers to make trainable
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Make top layers trainable
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
        
        # Add custom head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        logger.info(f"Created MobileNetV2 model with {model.count_params()} parameters")
        logger.info(f"Trainable parameters: {sum([np.prod(v.get_shape()) for v in model.trainable_weights])}")
        
        return model
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for MobileNetV2."""
        return {
            'trainable_layers': 20,
            'dropout_rate': 0.2
        }

class MLPFactory(ModelFactory):
    """Factory for Multi-Layer Perceptron models (tabular data)."""
    
    def create_model(self, 
                    input_shape: Tuple[int, ...], 
                    num_classes: int,
                    hidden_units: Tuple[int, ...] = (256, 128, 64),
                    dropout_rate: float = 0.3,
                    **kwargs) -> tf.keras.Model:
        """
        Create MLP model for tabular data.
        
        Args:
            input_shape: Input shape (features,)
            num_classes: Number of output classes
            hidden_units: Units for each hidden layer
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),
        ])
        
        # Add hidden layers
        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        logger.info(f"Created MLP model with {model.count_params()} parameters")
        return model
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for MLP."""
        return {
            'hidden_units': (256, 128, 64),
            'dropout_rate': 0.3
        }

# Model factory registry
MODEL_FACTORIES = {
    '1D-CNN': CNN1DFactory(),
    'BiLSTM': BiLSTMFactory(),
    '1D-CNN + BiLSTM': HybridCNNLSTMFactory(),
    'CNN-Small': SmallCNNFactory(),
    'MobileNetV2': MobileNetV2Factory(),
    'MLP': MLPFactory(),
    'Custom CNN': SmallCNNFactory(),  # Alias
}

def get_model_factory(model_type: str) -> ModelFactory:
    """
    Get model factory by type.
    
    Args:
        model_type: Type of model to create
        
    Returns:
        Model factory instance
        
    Raises:
        ValueError: If model type is not supported
    """
    if model_type not in MODEL_FACTORIES:
        available_types = list(MODEL_FACTORIES.keys())
        raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
    
    return MODEL_FACTORIES[model_type]

def list_available_models() -> List[str]:
    """List all available model types."""
    return list(MODEL_FACTORIES.keys())

def create_model(model_type: str, 
                input_shape: Tuple[int, ...], 
                num_classes: int,
                **kwargs) -> tf.keras.Model:
    """
    Convenience function to create a model.
    
    Args:
        model_type: Type of model to create
        input_shape: Input shape
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        Compiled Keras model
    """
    factory = get_model_factory(model_type)
    return factory.create_model(input_shape, num_classes, **kwargs)
