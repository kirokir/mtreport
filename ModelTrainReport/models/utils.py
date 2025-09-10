"""
Model Utilities
Common functions for model training, saving, and evaluation.
"""

import json
import os
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def save_model_config(model: tf.keras.Model, 
                     config: Dict[str, Any], 
                     save_dir: str) -> None:
    """
    Save model and its configuration.
    
    Args:
        model: Trained Keras model
        config: Training configuration
        save_dir: Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model.save(f"{save_dir}/model.h5")
    
    # Save model architecture as JSON
    with open(f"{save_dir}/model_architecture.json", "w") as f:
        f.write(model.to_json())
    
    # Save model summary
    with open(f"{save_dir}/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save configuration
    with open(f"{save_dir}/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model and config saved to {save_dir}")

def load_model_config(model_dir: str) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Load model and its configuration.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        Tuple of (model, config)
    """
    # Load model
    model_path = f"{model_dir}/model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = keras.models.load_model(model_path)
    
    # Load configuration
    config_path = f"{model_dir}/model_config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    logger.info(f"Model and config loaded from {model_dir}")
    return model, config

def create_callbacks(checkpoint_dir: str,
                    early_stopping: bool = True,
                    patience: int = 10,
                    monitor: str = 'val_loss',
                    reduce_lr: bool = True) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        monitor: Metric to monitor
        reduce_lr: Whether to reduce learning rate on plateau
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = f"{checkpoint_dir}/best_model.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode='min' if 'loss' in monitor else 'max',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if early_stopping:
        early_stop_callback = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='min' if 'loss' in monitor else 'max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop_callback)
    
    # Reduce learning rate on plateau
    if reduce_lr:
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=patience // 2,
            mode='min' if 'loss' in monitor else 'max',
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
    
    # CSV logger
    csv_logger = keras.callbacks.CSVLogger(
        f"{checkpoint_dir}/training_log.csv",
        append=True
    )
    callbacks.append(csv_logger)
    
    return callbacks

def evaluate_model(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Evaluate model and compute detailed metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Names of classes
        
    Returns:
        Dictionary of evaluation results
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Basic metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Detailed metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(np.unique(y_test)))]
    
    report = classification_report(y_test, y_pred, 
                                 target_names=class_names,
                                 output_dict=True)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'per_class_accuracy': per_class_accuracy.tolist(),
        'predictions': y_pred.tolist(),
        'prediction_probabilities': y_pred_proba.tolist()
    }
    
    return results

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def compute_model_size(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Compute model size metrics.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary of size metrics
    """
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Estimate model size in MB
    # Assuming float32 (4 bytes per parameter)
    size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(non_trainable_params),
        'size_mb': float(size_mb)
    }

def benchmark_inference_time(model: tf.keras.Model,
                           sample_input: np.ndarray,
                           num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark model inference time.
    
    Args:
        model: Keras model
        sample_input: Sample input for timing
        num_runs: Number of inference runs
        
    Returns:
        Dictionary of timing metrics
    """
    import time
    
    # Warm up
    for _ in range(10):
        _ = model.predict(sample_input, verbose=0)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(sample_input, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times))
    }

def analyze_model_errors(model: tf.keras.Model,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        class_names: List[str]) -> Dict[str, Any]:
    """
    Analyze model prediction errors.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Names of classes
        
    Returns:
        Dictionary of error analysis
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Find misclassified samples
    misclassified_idx = np.where(y_pred != y_test)[0]
    
    # Analyze confidence of wrong predictions
    wrong_confidences = []
    for idx in misclassified_idx:
        confidence = y_pred_proba[idx, y_pred[idx]]
        wrong_confidences.append(confidence)
    
    # Find most common confusion pairs
    confusion_pairs = {}
    for idx in misclassified_idx:
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        pair = f"{true_label} -> {pred_label}"
        
        if pair not in confusion_pairs:
            confusion_pairs[pair] = 0
        confusion_pairs[pair] += 1
    
    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    # Low confidence correct predictions
    correct_idx = np.where(y_pred == y_test)[0]
    correct_confidences = []
    for idx in correct_idx:
        confidence = y_pred_proba[idx, y_pred[idx]]
        correct_confidences.append(confidence)
    
    low_confidence_threshold = 0.6
    low_confidence_correct = np.sum(np.array(correct_confidences) < low_confidence_threshold)
    
    return {
        'total_errors': len(misclassified_idx),
        'error_rate': len(misclassified_idx) / len(y_test),
        'avg_wrong_confidence': np.mean(wrong_confidences) if wrong_confidences else 0,
        'top_confusion_pairs': sorted_pairs[:10],
        'low_confidence_correct_predictions': int(low_confidence_correct),
        'avg_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0
    }

def generate_model_summary_dict(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Generate a dictionary summary of the model.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model summary information
    """
    # Get layer information
    layers_info = []
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape),
            'param_count': layer.count_params()
        }
        layers_info.append(layer_info)
    
    # Model summary
    summary = {
        'total_layers': len(model.layers),
        'total_parameters': model.count_params(),
        'trainable_parameters': sum([np.prod(v.get_shape()) for v in model.trainable_weights]),
        'model_size_mb': model.count_params() * 4 / (1024 * 1024),
        'input_shape': str(model.input_shape),
        'output_shape': str(model.output_shape),
        'layers': layers_info
    }
    
    return summary
