"""
FastAPI Inference Server
Lightweight API for model predictions.
"""

import json
import os
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Local imports
from data_modules.skeleton_loader import validate_skeleton_schema, preprocess_skeleton_sequence
from data_modules.image_loader import preprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ML Training Platform API",
    description="Lightweight inference API for trained models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache: Dict[str, Dict[str, Any]] = {}

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    model_id: str
    data: Dict[str, Any]

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_id: str
    timestamp: str

class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    model_type: str
    dataset_type: str
    num_classes: int
    input_shape: List[int]
    accuracy: float
    timestamp: str

def load_model(model_id: str) -> Dict[str, Any]:
    """Load model and metadata."""
    if model_id in model_cache:
        return model_cache[model_id]
    
    model_dir = Path(f"checkpoints/{model_id}")
    
    if not model_dir.exists():
        raise ValueError(f"Model not found: {model_id}")
    
    # Load model
    model_path = model_dir / "model.h5"
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(str(model_path))
    
    # Load metadata
    config_path = model_dir / "run_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Load label encoder
    label_encoder_path = model_dir / "label_encoder.pkl"
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load results for accuracy info
    results_path = model_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)
    
    model_info = {
        'model': model,
        'config': config,
        'label_encoder': label_encoder,
        'results': results,
        'dataset_type': config.get('dataset_type', 'unknown'),
        'model_type': config['model'],
        'num_classes': len(label_encoder.classes_),
        'input_shape': results['dataset_info']['input_shape'],
        'accuracy': results['test_results']['accuracy']
    }
    
    model_cache[model_id] = model_info
    logger.info(f"Loaded model: {model_id}")
    
    return model_info

def preprocess_input_data(data: Dict[str, Any], model_info: Dict[str, Any]) -> np.ndarray:
    """Preprocess input data based on model type."""
    dataset_type = model_info['dataset_type']
    
    if dataset_type == 'skeleton':
        # Validate skeleton schema
        validation = validate_skeleton_schema(data)
        if not validation['valid']:
            raise ValueError(f"Invalid skeleton data: {validation['error']}")
        
        # Preprocess skeleton data
        processed_data = preprocess_skeleton_sequence(data)
        
        # Apply windowing (use same window length as training)
        window_length = model_info['config'].get('window_length', 64)
        
        if len(processed_data) < window_length:
            # Pad if sequence is too short
            padding = np.zeros((window_length - len(processed_data), processed_data.shape[1]))
            processed_data = np.vstack([processed_data, padding])
        elif len(processed_data) > window_length:
            # Take the last window
            processed_data = processed_data[-window_length:]
        
        return processed_data.reshape(1, window_length, -1)
    
    elif dataset_type == 'image':
        # Convert data to image array
        if 'image_array' in data:
            image_array = np.array(data['image_array'])
        else:
            raise ValueError("Image data must contain 'image_array' field")
        
        # Preprocess image
        target_size = model_info['input_shape'][:2]  # (height, width)
        processed_image = preprocess_image(image_array, target_size)
        
        return processed_image.reshape(1, *processed_image.shape)
    
    elif dataset_type == 'csv':
        # Extract features
        if 'features' in data:
            features = np.array(data['features'])
        else:
            raise ValueError("CSV data must contain 'features' field")
        
        return features.reshape(1, -1)
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ML Training Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    models = []
    
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                config_path = model_dir / "run_config.json"
                results_path = model_dir / "results.json"
                
                if config_path.exists() and results_path.exists():
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        
                        with open(results_path) as f:
                            results = json.load(f)
                        
                        models.append(ModelInfo(
                            model_id=config['run_id'],
                            model_type=config['model'],
                            dataset_type=config.get('dataset_type', 'unknown'),
                            num_classes=len(results.get('label_encoder_classes', [])),
                            input_shape=results['dataset_info']['input_shape'],
                            accuracy=results['test_results']['accuracy'],
                            timestamp=config['timestamp']
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to load model info for {model_dir.name}: {e}")
    
    return models

@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict(model_id: str, data: Dict[str, Any]):
    """Make prediction with specified model."""
    try:
        # Load model
        model_info = load_model(model_id)
        
        # Preprocess input
        processed_input = preprocess_input_data(data, model_info)
        
        # Make prediction
        predictions = model_info['model'].predict(processed_input)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        label_encoder = model_info['label_encoder']
        predicted_class = label_encoder.classes_[predicted_class_idx]
        
        # Create probability distribution
        probabilities = {}
        for i, class_name in enumerate(label_encoder.classes_):
            probabilities[class_name] = float(predictions[0][i])
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            model_id=model_id,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{model_id}/upload")
async def predict_upload(model_id: str, file: UploadFile = File(...)):
    """Make prediction with uploaded file."""
    try:
        # Load model
        model_info = load_model(model_id)
        dataset_type = model_info['dataset_type']
        
        # Process uploaded file
        content = await file.read()
        
        if dataset_type == 'skeleton':
            if not file.filename.endswith('.json'):
                raise ValueError("Skeleton models require JSON files")
            
            data = json.loads(content.decode())
            
        elif dataset_type == 'image':
            if not any(file.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                raise ValueError("Image models require image files")
            
            # Convert image to array
            import cv2
            import io
            
            image_bytes = io.BytesIO(content)
            image_array = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
            
            data = {'image_array': image_array.tolist()}
            
        else:
            raise ValueError(f"File upload not supported for dataset type: {dataset_type}")
        
        # Make prediction using the data endpoint
        return await predict(model_id, data)
    
    except Exception as e:
        logger.error(f"Upload prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get detailed model information."""
    try:
        model_info = load_model(model_id)
        
        return {
            'model_id': model_id,
            'config': model_info['config'],
            'results': model_info['results'],
            'classes': model_info['label_encoder'].classes_.tolist(),
            'model_summary': str(model_info['model'].to_json())
        }
    
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "predict_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
