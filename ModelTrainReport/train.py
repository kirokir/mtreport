"""
CLI Training Script
Handles model training with MLflow tracking and checkpointing.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import logging

# Local imports
from data_modules.image_loader import ImageDataLoader
from data_modules.skeleton_loader import SkeletonDataLoader
from data_modules.csv_loader import CSVDataLoader
from models.architectures import get_model_factory
from models.utils import save_model_config, create_callbacks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_mlflow_available() -> bool:
    """Check if MLflow is available."""
    try:
        import mlflow
        return True
    except ImportError:
        return False

def setup_mlflow(config: Dict[str, Any]) -> Optional[Any]:
    """Setup MLflow tracking if available."""
    if not config.get('use_mlflow', False) or not check_mlflow_available():
        logger.info("MLflow not available or disabled. Using JSON logging.")
        return None
    
    try:
        import mlflow
        import mlflow.tensorflow
        
        mlflow.set_tracking_uri("./mlruns")
        
        # Start MLflow run
        mlflow.start_run(run_name=config.get('run_label', config['run_id']))
        
        # Log parameters
        mlflow.log_params({
            'model_architecture': config['model'],
            'dataset': config['dataset'],
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'git_commit': config.get('git_commit', 'unknown')
        })
        
        return mlflow
    
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Falling back to JSON logging.")
        return None

def load_dataset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess dataset based on configuration."""
    dataset_name = config['dataset']
    
    # Load dataset metadata
    datasets_file = "datasets.json"
    if os.path.exists(datasets_file):
        with open(datasets_file) as f:
            datasets = json.load(f)
    else:
        raise ValueError(f"Dataset metadata not found: {datasets_file}")
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset not found: {dataset_name}")
    
    dataset_info = datasets[dataset_name]
    dataset_type = dataset_info['type']
    
    if dataset_type == 'skeleton':
        loader = SkeletonDataLoader()
        X, y, metadata = loader.load_processed_data(dataset_info['data_path'])
        
        # Apply windowing for skeleton data
        window_length = config.get('window_length', 64)
        stride = config.get('stride', window_length // 2)
        
        X_windowed = []
        y_windowed = []
        
        for seq_idx, (seq_data, seq_labels) in enumerate(zip(X, y)):
            for start_idx in range(0, len(seq_data) - window_length + 1, stride):
                window = seq_data[start_idx:start_idx + window_length]
                # Use majority label for window
                window_labels = seq_labels[start_idx:start_idx + window_length]
                majority_label = np.bincount(window_labels).argmax()
                
                X_windowed.append(window)
                y_windowed.append(majority_label)
        
        X = np.array(X_windowed)
        y = np.array(y_windowed)
    
    elif dataset_type == 'image':
        loader = ImageDataLoader()
        X, y, metadata = loader.load_processed_data(dataset_info['data_path'])
    
    elif dataset_type == 'csv':
        loader = CSVDataLoader()
        X, y, metadata = loader.load_processed_data(dataset_info['data_path'])
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return X, y, metadata

def create_model(config: Dict[str, Any], input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    """Create model based on configuration."""
    model_type = config['model']
    
    factory = get_model_factory(model_type)
    model = factory.create_model(
        input_shape=input_shape,
        num_classes=num_classes,
        **config.get('model_params', {})
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main training function."""
    logger.info(f"Starting training for run: {config['run_id']}")
    
    # Setup MLflow
    mlflow = setup_mlflow(config)
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        X, y, metadata = load_dataset(config)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        
        logger.info(f"Dataset loaded: {X.shape} samples, {num_classes} classes")
        
        # Train/validation/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config, X_train.shape[1:], num_classes)
        
        logger.info(f"Model created: {model.count_params()} parameters")
        model.summary(print_fn=logger.info)
        
        # Setup callbacks
        checkpoint_dir = f"checkpoints/{config['run_id']}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = create_callbacks(
            checkpoint_dir=checkpoint_dir,
            early_stopping=config.get('early_stopping', True),
            patience=10
        )
        
        # Train model
        logger.info("Starting training...")
        start_time = datetime.now()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Generate predictions for detailed metrics
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        cm = confusion_matrix(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes, 
                                     target_names=label_encoder.classes_, 
                                     output_dict=True)
        
        # Prepare results
        results = {
            'run_id': config['run_id'],
            'config': config,
            'training_time': training_time,
            'history': {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'val_loss': history.history['val_loss'],
                'val_accuracy': history.history['val_accuracy']
            },
            'test_results': {
                'loss': float(test_loss),
                'accuracy': float(test_accuracy),
                'f1_score': float(f1)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'label_encoder_classes': label_encoder.classes_.tolist(),
            'model_params': model.count_params(),
            'dataset_info': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'num_classes': num_classes,
                'input_shape': X_train.shape[1:]
            }
        }
        
        # Log to MLflow
        if mlflow:
            mlflow.log_metrics({
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'test_f1_score': f1,
                'training_time': training_time,
                'model_params': model.count_params()
            })
            
            # Log final model
            mlflow.tensorflow.log_model(model, "model")
        
        # Save results
        model.save(f"{checkpoint_dir}/model.h5")
        
        with open(f"{checkpoint_dir}/history.pkl", "wb") as f:
            pickle.dump(results, f)
        
        with open(f"{checkpoint_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save label encoder
        with open(f"{checkpoint_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if mlflow:
            mlflow.log_param("error", str(e))
        raise
    
    finally:
        if mlflow:
            try:
                import mlflow
                mlflow.end_run()
            except:
                pass

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--gpu", type=int, help="GPU device to use")
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load configuration
    with open(args.config) as f:
        config = json.load(f)
    
    # Run training
    try:
        results = train_model(config)
        logger.info("Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
