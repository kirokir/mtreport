"""
Image Data Loader
Handles loading and preprocessing of image datasets.
"""

import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class ImageDataLoader:
    """Loader for image datasets."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image loader.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    def load_from_uploaded_files(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Load images from uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Dataset information dictionary
        """
        images = []
        labels = []
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                
                if filename.endswith('.zip'):
                    # Extract zip file
                    zip_path = os.path.join(temp_dir, filename)
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Process extracted images
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in self.supported_formats):
                                file_path = os.path.join(root, file)
                                
                                # Use parent directory as label
                                label = os.path.basename(root)
                                if label == os.path.basename(temp_dir):
                                    label = "unlabeled"
                                
                                try:
                                    image = self._load_and_preprocess_image(file_path)
                                    images.append(image)
                                    labels.append(label)
                                except Exception as e:
                                    logger.warning(f"Failed to load image {file_path}: {e}")
                
                else:
                    # Single image file
                    if any(filename.lower().endswith(ext) for ext in self.supported_formats):
                        # Save uploaded file temporarily
                        temp_path = os.path.join(temp_dir, filename)
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.read())
                        
                        try:
                            image = self._load_and_preprocess_image(temp_path)
                            images.append(image)
                            
                            # Extract label from filename
                            label = Path(filename).stem
                            labels.append(label)
                        except Exception as e:
                            logger.warning(f"Failed to load image {filename}: {e}")
            
            if not images:
                raise ValueError("No valid images found")
            
            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(labels)
            
            # Get class distribution
            unique_labels, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique_labels, counts.astype(int)))
            
            # Save processed data
            dataset_id = f"images_{len(images)}_{len(unique_labels)}"
            save_dir = f"data/{dataset_id}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save images and metadata
            np.save(f"{save_dir}/images.npy", X)
            np.save(f"{save_dir}/labels.npy", y)
            
            # Sample images for preview
            sample_images = []
            for label in unique_labels[:4]:  # Max 4 classes for preview
                label_indices = np.where(y == label)[0]
                if len(label_indices) > 0:
                    sample_idx = label_indices[0]
                    sample_path = f"{save_dir}/sample_{sample_idx}.png"
                    
                    # Save sample image
                    sample_image = (X[sample_idx] * 255).astype(np.uint8)
                    Image.fromarray(sample_image).save(sample_path)
                    sample_images.append((sample_path, label))
            
            return {
                'type': 'image',
                'dataset_id': dataset_id,
                'data_path': save_dir,
                'num_samples': len(images),
                'num_classes': len(unique_labels),
                'image_shape': X[0].shape,
                'class_distribution': class_distribution,
                'sample_images': sample_images
            }
        
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            return image_array
        
        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {e}")
    
    def load_processed_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load processed data from disk.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Tuple of (X, y, metadata)
        """
        X = np.load(f"{data_path}/images.npy")
        y = np.load(f"{data_path}/labels.npy")
        
        metadata = {
            'num_samples': len(X),
            'image_shape': X[0].shape,
            'num_classes': len(np.unique(y))
        }
        
        return X, y, metadata
    
    def analyze_dataset(self, data_path: str) -> Dict[str, Any]:
        """
        Analyze dataset and compute rich statistics.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Analysis results
        """
        X, y, _ = self.load_processed_data(data_path)
        
        # Basic statistics
        stats = {
            'num_samples': len(X),
            'image_shape': X[0].shape,
            'num_classes': len(np.unique(y))
        }
        
        # Pixel intensity statistics
        if len(X.shape) == 4:  # (samples, height, width, channels)
            # Per-channel statistics
            channel_stats = {}
            for i in range(X.shape[-1]):
                channel_data = X[:, :, :, i]
                channel_stats[f'channel_{i}'] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': float(np.min(channel_data)),
                    'max': float(np.max(channel_data))
                }
            
            stats['channel_statistics'] = channel_stats
            
            # Overall intensity histogram
            all_pixels = X.flatten()
            hist, bin_edges = np.histogram(all_pixels, bins=50)
            stats['intensity_histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        
        # Class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        stats['class_distribution'] = dict(zip(unique_labels, counts.astype(int)))
        
        # Missing/corrupted image detection
        corrupted_indices = []
        for i, img in enumerate(X):
            # Check for NaN or infinite values
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                corrupted_indices.append(i)
            
            # Check for completely black or white images
            if np.all(img == 0) or np.all(img == 1):
                corrupted_indices.append(i)
        
        stats['corrupted_images'] = {
            'count': len(corrupted_indices),
            'indices': corrupted_indices
        }
        
        return stats

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image for inference.
    
    Args:
        image: Input image array
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image
    """
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize image
    resized = cv2.resize(image, (target_size[1], target_size[0]))
    
    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized
