"""
Tests for data loaders.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from data_modules.image_loader import ImageDataLoader
from data_modules.skeleton_loader import SkeletonDataLoader, validate_skeleton_schema
from data_modules.csv_loader import CSVDataLoader

class TestImageLoader(unittest.TestCase):
    """Test cases for ImageDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ImageDataLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_and_preprocess_image(self):
        """Test image loading and preprocessing."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        image_path = os.path.join(self.temp_dir, 'test_image.png')
        test_image.save(image_path)
        
        # Test loading
        processed_image = self.loader._load_and_preprocess_image(image_path)
        
        # Check output shape and type
        self.assertEqual(processed_image.shape, (224, 224, 3))
        self.assertEqual(processed_image.dtype, np.float32)
        self.assertTrue(0 <= processed_image.min() <= processed_image.max() <= 1)
    
    def test_invalid_image_file(self):
        """Test handling of invalid image files."""
        invalid_path = os.path.join(self.temp_dir, 'invalid.txt')
        with open(invalid_path, 'w') as f:
            f.write("This is not an image")
        
        with self.assertRaises(ValueError):
            self.loader._load_and_preprocess_image(invalid_path)
    
    def test_analyze_dataset(self):
        """Test dataset analysis functionality."""
        # Create mock dataset
        data_dir = os.path.join(self.temp_dir, 'test_dataset')
        os.makedirs(data_dir)
        
        # Create mock data files
        images = np.random.random((10, 224, 224, 3)).astype(np.float32)
        labels = np.array(['class_0', 'class_1'] * 5)
        
        np.save(os.path.join(data_dir, 'images.npy'), images)
        np.save(os.path.join(data_dir, 'labels.npy'), labels)
        
        # Test analysis
        stats = self.loader.analyze_dataset(data_dir)
        
        # Check results
        self.assertEqual(stats['num_samples'], 10)
        self.assertEqual(stats['image_shape'], (224, 224, 3))
        self.assertEqual(stats['num_classes'], 2)
        self.assertIn('channel_statistics', stats)
        self.assertIn('intensity_histogram', stats)

class TestSkeletonLoader(unittest.TestCase):
    """Test cases for SkeletonDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = SkeletonDataLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_valid_skeleton_data(self):
        """Create valid skeleton data for testing."""
        return {
            "id": "test_sequence",
            "fps": 30,
            "frames": [
                {
                    "t": 0.0,
                    "joints": {
                        "wrist": [0.1, 0.2, 0.3],
                        "index": [0.4, 0.5, 0.6],
                        "thumb": [0.7, 0.8, 0.9]
                    },
                    "label": "gesture_a"
                },
                {
                    "t": 0.033,
                    "joints": {
                        "wrist": [0.11, 0.21, 0.31],
                        "index": [0.41, 0.51, 0.61],
                        "thumb": [0.71, 0.81, 0.91]
                    },
                    "label": "gesture_a"
                }
            ]
        }
    
    def test_validate_skeleton_schema_valid(self):
        """Test validation of valid skeleton data."""
        data = self.create_valid_skeleton_data()
        result = validate_skeleton_schema(data)
        
        self.assertTrue(result['valid'])
        self.assertIsNone(result['error'])
    
    def test_validate_skeleton_schema_missing_id(self):
        """Test validation with missing ID."""
        data = self.create_valid_skeleton_data()
        del data['id']
        
        result = validate_skeleton_schema(data)
        
        self.assertFalse(result['valid'])
        self.assertIn('Missing required field: id', result['error'])
    
    def test_validate_skeleton_schema_invalid_fps(self):
        """Test validation with invalid FPS."""
        data = self.create_valid_skeleton_data()
        data['fps'] = -1
        
        result = validate_skeleton_schema(data)
        
        self.assertFalse(result['valid'])
        self.assertIn('fps must be a positive number', result['error'])
    
    def test_validate_skeleton_schema_inconsistent_joints(self):
        """Test validation with inconsistent joints."""
        data = self.create_valid_skeleton_data()
        # Remove a joint from second frame
        del data['frames'][1]['joints']['thumb']
        
        result = validate_skeleton_schema(data)
        
        self.assertFalse(result['valid'])
        self.assertIn('inconsistent joint keys', result['error'])
    
    def test_extract_features_from_sequence(self):
        """Test feature extraction from skeleton sequence."""
        from data_modules.skeleton_loader import extract_features_from_sequence
        
        data = self.create_valid_skeleton_data()
        features = extract_features_from_sequence(data)
        
        # Check output shape: 2 frames, 3 joints * 3 coords = 9 features per frame
        # Plus velocities = 18 features total
        self.assertEqual(features.shape, (2, 18))
        self.assertEqual(features.dtype, np.float64)

class TestCSVLoader(unittest.TestCase):
    """Test cases for CSVDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = CSVDataLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_csv(self):
        """Create a test CSV file."""
        data = {
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature_3': ['a', 'b', 'a', 'b', 'a'],
            'target': ['class_0', 'class_1', 'class_0', 'class_1', 'class_0']
        }
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_load_from_uploaded_file(self):
        """Test loading CSV from uploaded file."""
        csv_path = self.create_test_csv()
        
        # Mock uploaded file
        with open(csv_path, 'rb') as f:
            mock_file = Mock()
            mock_file.read.return_value = f.read()
        
        # Reset file position for pandas
        mock_file.seek = Mock()
        with open(csv_path, 'r') as f:
            content = f.read()
        
        # Mock pandas read_csv to use our content
        with patch('pandas.read_csv') as mock_read_csv:
            df = pd.DataFrame({
                'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
                'feature_3': [0, 1, 0, 1, 0],  # Encoded categorical
                'target': ['class_0', 'class_1', 'class_0', 'class_1', 'class_0']
            })
            mock_read_csv.return_value = df
            
            result = self.loader.load_from_uploaded_file(mock_file)
            
            # Check results
            self.assertEqual(result['type'], 'csv')
            self.assertEqual(result['num_samples'], 5)
            self.assertEqual(result['num_features'], 3)
    
    def test_compute_feature_statistics(self):
        """Test computation of feature statistics."""
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature_2': [0.1, 0.2, np.nan, 0.4, 0.5],
            'target': ['class_0', 'class_1', 'class_0', 'class_1', 'class_0']
        })
        
        X = df[['feature_1', 'feature_2']]
        y = df['target']
        
        stats = self.loader._compute_feature_statistics(X, y)
        
        # Check basic stats
        self.assertEqual(stats['basic']['num_features'], 2)
        self.assertEqual(stats['basic']['num_samples'], 5)
        self.assertEqual(stats['basic']['missing_values_per_column']['feature_2'], 1)
        
        # Check numeric feature stats
        self.assertIn('numeric_features', stats)
        self.assertIn('feature_1', stats['numeric_features'])
        self.assertEqual(stats['numeric_features']['feature_1']['mean'], 3.0)

class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests for data loaders."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test end-to-end data loading workflow."""
        # Test skeleton data workflow
        skeleton_loader = SkeletonDataLoader()
        
        # Create test skeleton file
        skeleton_data = {
            "id": "test_seq",
            "fps": 30,
            "frames": [
                {
                    "t": i * 0.033,
                    "joints": {
                        "wrist": [i * 0.1, i * 0.1, i * 0.1],
                        "index": [i * 0.2, i * 0.2, i * 0.2]
                    },
                    "label": f"gesture_{i % 2}"
                }
                for i in range(10)
            ]
        }
        
        skeleton_file = os.path.join(self.temp_dir, 'test_skeleton.json')
        with open(skeleton_file, 'w') as f:
            json.dump(skeleton_data, f)
        
        # Mock uploaded file
        with open(skeleton_file, 'r') as f:
            content = f.read()
        
        mock_file = Mock()
        mock_file.name = 'test_skeleton.json'
        mock_file.read.return_value = content.encode()
        mock_file.seek = Mock()
        
        # Test loading
        with patch('tempfile.mkdtemp', return_value=self.temp_dir):
            with patch('os.makedirs'):
                with patch('json.dump'):
                    result = skeleton_loader.load_from_uploaded_files([mock_file])
        
        # Check results
        self.assertEqual(result['type'], 'skeleton')
        self.assertEqual(result['num_sequences'], 1)

if __name__ == '__main__':
    unittest.main()
