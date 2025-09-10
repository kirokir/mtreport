"""
Skeleton Data Loader
Handles loading and preprocessing of skeleton JSON data.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

def validate_skeleton_schema(data: Dict[str, Any]) -> Dict[str, Union[bool, str]]:
    """
    Validate skeleton JSON schema.
    
    Args:
        data: Skeleton data dictionary
        
    Returns:
        Validation result with 'valid' boolean and 'error' message
    """
    try:
        # Check required fields
        if 'id' not in data:
            return {'valid': False, 'error': 'Missing required field: id'}
        
        if 'fps' not in data:
            return {'valid': False, 'error': 'Missing required field: fps'}
        
        if not isinstance(data['fps'], (int, float)) or data['fps'] <= 0:
            return {'valid': False, 'error': 'fps must be a positive number'}
        
        if 'frames' not in data:
            return {'valid': False, 'error': 'Missing required field: frames'}
        
        if not isinstance(data['frames'], list) or len(data['frames']) == 0:
            return {'valid': False, 'error': 'frames must be a non-empty list'}
        
        # Check frame structure
        joint_keys = None
        for i, frame in enumerate(data['frames']):
            # Check time field
            if 't' not in frame:
                return {'valid': False, 'error': f'Frame {i} missing time field: t'}
            
            if not isinstance(frame['t'], (int, float)):
                return {'valid': False, 'error': f'Frame {i} time field must be numeric'}
            
            # Check joints
            if 'joints' not in frame:
                return {'valid': False, 'error': f'Frame {i} missing joints field'}
            
            if not isinstance(frame['joints'], dict):
                return {'valid': False, 'error': f'Frame {i} joints must be a dictionary'}
            
            # Check joint consistency
            current_joints = set(frame['joints'].keys())
            if joint_keys is None:
                joint_keys = current_joints
            elif current_joints != joint_keys:
                return {'valid': False, 'error': f'Frame {i} has inconsistent joint keys'}
            
            # Check joint coordinates
            for joint_name, coords in frame['joints'].items():
                if not isinstance(coords, list) or len(coords) != 3:
                    return {'valid': False, 'error': f'Frame {i} joint {joint_name} must be [x, y, z]'}
                
                if not all(isinstance(c, (int, float)) for c in coords):
                    return {'valid': False, 'error': f'Frame {i} joint {joint_name} coordinates must be numeric'}
            
            # Check label (optional)
            if 'label' in frame and not isinstance(frame['label'], str):
                return {'valid': False, 'error': f'Frame {i} label must be a string'}
        
        return {'valid': True, 'error': None}
    
    except Exception as e:
        return {'valid': False, 'error': f'Validation error: {str(e)}'}

def normalize_skeleton_sequence(sequence_data: Dict[str, Any], 
                               root_joint: str = 'wrist',
                               scale_normalize: bool = True) -> Dict[str, Any]:
    """
    Normalize skeleton sequence.
    
    Args:
        sequence_data: Raw skeleton sequence
        root_joint: Joint to use as root for normalization
        scale_normalize: Whether to apply scale normalization
        
    Returns:
        Normalized sequence data
    """
    frames = sequence_data['frames']
    normalized_frames = []
    
    for frame in frames:
        joints = frame['joints'].copy()
        
        # Root normalization
        if root_joint in joints:
            root_pos = np.array(joints[root_joint])
            
            # Subtract root position from all joints
            for joint_name in joints:
                joints[joint_name] = (np.array(joints[joint_name]) - root_pos).tolist()
        
        # Scale normalization
        if scale_normalize:
            # Compute skeleton height (max pairwise distance)
            joint_positions = np.array(list(joints.values()))
            if len(joint_positions) > 1:
                distances = []
                for i in range(len(joint_positions)):
                    for j in range(i + 1, len(joint_positions)):
                        dist = np.linalg.norm(joint_positions[i] - joint_positions[j])
                        distances.append(dist)
                
                if distances:
                    skeleton_height = max(distances)
                    
                    # Scale all joints
                    if skeleton_height > 1e-6:
                        for joint_name in joints:
                            joints[joint_name] = (np.array(joints[joint_name]) / skeleton_height).tolist()
        
        normalized_frame = frame.copy()
        normalized_frame['joints'] = joints
        normalized_frames.append(normalized_frame)
    
    normalized_sequence = sequence_data.copy()
    normalized_sequence['frames'] = normalized_frames
    
    return normalized_sequence

def interpolate_missing_joints(sequence_data: Dict[str, Any], 
                              max_missing_ratio: float = 0.3) -> Dict[str, Any]:
    """
    Interpolate missing joint data.
    
    Args:
        sequence_data: Skeleton sequence data
        max_missing_ratio: Maximum ratio of missing frames allowed per joint
        
    Returns:
        Sequence with interpolated joints
    """
    frames = sequence_data['frames']
    
    if len(frames) < 3:  # Need at least 3 frames for interpolation
        return sequence_data
    
    # Get all joint names
    joint_names = set()
    for frame in frames:
        joint_names.update(frame['joints'].keys())
    
    joint_names = list(joint_names)
    
    # Create time array
    times = np.array([frame['t'] for frame in frames])
    
    # Process each joint
    for joint_name in joint_names:
        # Extract joint coordinates across all frames
        x_coords = []
        y_coords = []
        z_coords = []
        valid_indices = []
        
        for i, frame in enumerate(frames):
            if joint_name in frame['joints']:
                coords = frame['joints'][joint_name]
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                z_coords.append(coords[2])
                valid_indices.append(i)
        
        # Check if joint has too many missing values
        missing_ratio = 1.0 - len(valid_indices) / len(frames)
        if missing_ratio > max_missing_ratio:
            logger.warning(f"Joint {joint_name} has {missing_ratio:.1%} missing data - removing joint")
            # Remove joint from all frames
            for frame in frames:
                if joint_name in frame['joints']:
                    del frame['joints'][joint_name]
            continue
        
        # Interpolate missing values
        if len(valid_indices) > 1 and len(valid_indices) < len(frames):
            valid_times = times[valid_indices]
            
            # Create interpolation functions
            interp_x = interp1d(valid_times, x_coords, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
            interp_y = interp1d(valid_times, y_coords, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
            interp_z = interp1d(valid_times, z_coords, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
            
            # Fill missing values
            for i, frame in enumerate(frames):
                if joint_name not in frame['joints']:
                    interpolated_coords = [
                        float(interp_x(times[i])),
                        float(interp_y(times[i])),
                        float(interp_z(times[i]))
                    ]
                    frame['joints'][joint_name] = interpolated_coords
    
    return sequence_data

def extract_features_from_sequence(sequence_data: Dict[str, Any], 
                                  include_velocities: bool = True,
                                  include_accelerations: bool = False) -> np.ndarray:
    """
    Extract features from skeleton sequence.
    
    Args:
        sequence_data: Normalized skeleton sequence
        include_velocities: Whether to include velocity features
        include_accelerations: Whether to include acceleration features
        
    Returns:
        Feature array of shape (n_frames, n_features)
    """
    frames = sequence_data['frames']
    
    if not frames:
        return np.array([])
    
    # Get joint names (consistent across frames)
    joint_names = sorted(frames[0]['joints'].keys())
    n_joints = len(joint_names)
    
    # Extract positions
    positions = []
    for frame in frames:
        frame_positions = []
        for joint_name in joint_names:
            coords = frame['joints'][joint_name]
            frame_positions.extend(coords)  # Flatten [x, y, z]
        positions.append(frame_positions)
    
    positions = np.array(positions)  # Shape: (n_frames, n_joints * 3)
    
    features = [positions]
    
    # Compute velocities
    if include_velocities and len(positions) > 1:
        velocities = np.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        features.append(velocities)
    
    # Compute accelerations
    if include_accelerations and len(positions) > 2:
        accelerations = np.zeros_like(positions)
        accelerations[2:] = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        features.append(accelerations)
    
    # Concatenate all features
    features_array = np.concatenate(features, axis=1)
    
    return features_array

class SkeletonDataLoader:
    """Loader for skeleton JSON datasets."""
    
    def __init__(self, 
                 root_joint: str = 'wrist',
                 scale_normalize: bool = True,
                 interpolate_missing: bool = True):
        """
        Initialize skeleton loader.
        
        Args:
            root_joint: Joint to use as root for normalization
            scale_normalize: Whether to apply scale normalization
            interpolate_missing: Whether to interpolate missing joint data
        """
        self.root_joint = root_joint
        self.scale_normalize = scale_normalize
        self.interpolate_missing = interpolate_missing
    
    def load_from_uploaded_files(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Load skeleton data from uploaded JSON files.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Dataset information dictionary
        """
        sequences = []
        sequence_lengths = []
        all_labels = []
        
        for uploaded_file in uploaded_files:
            try:
                content = json.loads(uploaded_file.read().decode())
                uploaded_file.seek(0)  # Reset file pointer
                
                # Validate schema
                validation = validate_skeleton_schema(content)
                if not validation['valid']:
                    logger.error(f"Invalid file {uploaded_file.name}: {validation['error']}")
                    continue
                
                # Normalize sequence
                if self.scale_normalize or self.root_joint:
                    content = normalize_skeleton_sequence(
                        content, 
                        root_joint=self.root_joint,
                        scale_normalize=self.scale_normalize
                    )
                
                # Interpolate missing joints
                if self.interpolate_missing:
                    content = interpolate_missing_joints(content)
                
                sequences.append(content)
                sequence_lengths.append(len(content['frames']))
                
                # Extract labels
                frame_labels = []
                for frame in content['frames']:
                    label = frame.get('label', 'unlabeled')
                    frame_labels.append(label)
                    all_labels.append(label)
                
            except Exception as e:
                logger.error(f"Failed to process file {uploaded_file.name}: {e}")
        
        if not sequences:
            raise ValueError("No valid skeleton sequences found")
        
        # Get class distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts.astype(int)))
        
        # Save processed data
        dataset_id = f"skeleton_{len(sequences)}_{len(unique_labels)}"
        save_dir = f"data/{dataset_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save sequences
        with open(f"{save_dir}/sequences.json", "w") as f:
            json.dump(sequences, f, indent=2)
        
        return {
            'type': 'skeleton',
            'dataset_id': dataset_id,
            'data_path': save_dir,
            'num_sequences': len(sequences),
            'num_classes': len(unique_labels),
            'class_distribution': class_distribution,
            'sequence_lengths': sequence_lengths,
            'sample_sequence': sequences[0] if sequences else None
        }
    
    def load_processed_data(self, data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
        """
        Load processed skeleton data from disk.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Tuple of (X_sequences, y_sequences, metadata)
        """
        # Load sequences
        with open(f"{data_path}/sequences.json") as f:
            sequences = json.load(f)
        
        X_sequences = []
        y_sequences = []
        
        for sequence in sequences:
            # Extract features
            features = extract_features_from_sequence(sequence)
            X_sequences.append(features)
            
            # Extract labels
            labels = []
            for frame in sequence['frames']:
                label = frame.get('label', 'unlabeled')
                labels.append(label)
            y_sequences.append(np.array(labels))
        
        metadata = {
            'num_sequences': len(sequences),
            'total_frames': sum(len(seq['frames']) for seq in sequences),
            'feature_dim': X_sequences[0].shape[1] if X_sequences else 0
        }
        
        return X_sequences, y_sequences, metadata
    
    def analyze_dataset(self, data_path: str) -> Dict[str, Any]:
        """
        Analyze skeleton dataset and compute rich statistics.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Analysis results
        """
        with open(f"{data_path}/sequences.json") as f:
            sequences = json.load(f)
        
        # Basic statistics
        sequence_lengths = [len(seq['frames']) for seq in sequences]
        stats = {
            'num_sequences': len(sequences),
            'total_frames': sum(sequence_lengths),
            'sequence_lengths': sequence_lengths,
            'avg_sequence_length': np.mean(sequence_lengths),
            'min_sequence_length': min(sequence_lengths),
            'max_sequence_length': max(sequence_lengths)
        }
        
        # Get all joint names
        joint_names = set()
        for sequence in sequences:
            for frame in sequence['frames']:
                joint_names.update(frame['joints'].keys())
        
        joint_names = sorted(joint_names)
        stats['joint_names'] = joint_names
        stats['num_joints'] = len(joint_names)
        
        # Per-joint statistics
        joint_stats = {}
        for joint_name in joint_names:
            all_positions = []
            
            for sequence in sequences:
                for frame in sequence['frames']:
                    if joint_name in frame['joints']:
                        coords = frame['joints'][joint_name]
                        all_positions.append(coords)
            
            if all_positions:
                all_positions = np.array(all_positions)
                
                joint_stats[joint_name] = {
                    'mean': np.mean(all_positions, axis=0).tolist(),
                    'std': np.std(all_positions, axis=0).tolist(),
                    'min': np.min(all_positions, axis=0).tolist(),
                    'max': np.max(all_positions, axis=0).tolist(),
                    'motion_range': (np.max(all_positions, axis=0) - np.min(all_positions, axis=0)).tolist()
                }
        
        stats['joint_statistics'] = joint_stats
        
        # Label distribution
        all_labels = []
        for sequence in sequences:
            for frame in sequence['frames']:
                label = frame.get('label', 'unlabeled')
                all_labels.append(label)
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        stats['class_distribution'] = dict(zip(unique_labels, counts.astype(int)))
        
        return stats

def preprocess_skeleton_sequence(sequence_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess a single skeleton sequence for inference.
    
    Args:
        sequence_data: Raw skeleton sequence
        
    Returns:
        Preprocessed feature array
    """
    # Normalize sequence
    normalized = normalize_skeleton_sequence(sequence_data)
    
    # Interpolate missing joints
    interpolated = interpolate_missing_joints(normalized)
    
    # Extract features
    features = extract_features_from_sequence(interpolated)
    
    return features
