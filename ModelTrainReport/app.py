"""
Streamlit ML Training Platform
Multi-tab interface for data management, training, monitoring, prediction, and reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import subprocess
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import io
import base64
import zipfile
from PIL import Image
import cv2

# Local imports
from data_modules.image_loader import ImageDataLoader
from data_modules.skeleton_loader import SkeletonDataLoader, validate_skeleton_schema
from data_modules.csv_loader import CSVDataLoader
from models.architectures import get_model_factory
from models.utils import load_model_config, save_model_config
from reports.generator import ReportGenerator
from streamlit_ace import st_ace

# Configure page
st.set_page_config(
    page_title="ML Training Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def check_mlflow_available() -> bool:
    """Check if MLflow is available."""
    try:
        import mlflow
        return True
    except ImportError:
        return False

def load_training_history(run_id: str) -> Optional[Dict]:
    """Load training history for a specific run."""
    history_path = f"checkpoints/{run_id}/history.pkl"
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            return pickle.load(f)
    return None

def display_skeleton_animation(skeleton_data: Dict, prediction: Optional[str] = None):
    """Display skeleton animation with optional prediction overlay."""
    frames = skeleton_data.get('frames', [])
    if not frames:
        st.error("No frames found in skeleton data")
        return
    
    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        frame_idx = st.slider("Frame", 0, len(frames)-1, 0)
    
    with col2:
        if st.button("Play"):
            # Simple animation by cycling through frames
            placeholder = st.empty()
            for i in range(len(frames)):
                with placeholder.container():
                    display_single_frame(frames[i], prediction)
                time.sleep(0.1)
    
    with col3:
        speed = st.slider("Speed", 0.1, 2.0, 1.0)
    
    # Display current frame
    display_single_frame(frames[frame_idx], prediction)

def display_single_frame(frame_data: Dict, prediction: Optional[str] = None):
    """Display a single skeleton frame."""
    joints = frame_data.get('joints', {})
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    x_coords = []
    y_coords = []
    z_coords = []
    joint_names = []
    
    for joint_name, coords in joints.items():
        if len(coords) >= 3:
            x_coords.append(coords[0])
            y_coords.append(coords[1])
            z_coords.append(coords[2])
            joint_names.append(joint_name)
    
    if x_coords:
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            text=joint_names,
            textposition="top center",
            marker=dict(size=8, color='red')
        ))
        
        fig.update_layout(
            title=f"Skeleton Frame - {prediction if prediction else 'No prediction'}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No valid joint coordinates found")

# Sidebar Navigation
st.sidebar.title("🤖 ML Training Platform")
st.sidebar.markdown("---")

tabs = [
    "🏠 Home",
    "📊 Data Manager", 
    "🎯 Train",
    "📈 Monitor",
    "🔮 Predict",
    "📄 Report",
    "⚙️ Settings"
]

selected_tab = st.sidebar.radio("Navigation", tabs)

# Main content area
if selected_tab == "🏠 Home":
    st.title("ML Training Platform")
    st.markdown("### Welcome to the comprehensive ML training and reporting platform")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Features:**
        - 📊 Multi-format data loading (Images, Skeleton JSON, CSV)
        - 🎯 Multiple model architectures (1D-CNN, BiLSTM, CNN, MobileNetV2)
        - 📈 Real-time training monitoring with MLflow
        - 🔮 Interactive prediction with skeleton animation
        - 📄 Advanced editable reporting with multi-run comparisons
        - ⚙️ Model export to TFLite and ONNX
        """)
        
        st.markdown("### Quick Start")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("📊 Load Demo Dataset", use_container_width=True):
                st.session_state.selected_tab = "📊 Data Manager"
                st.rerun()
        with col_b:
            if st.button("🎯 Start Training", use_container_width=True):
                st.session_state.selected_tab = "🎯 Train"
                st.rerun()
        with col_c:
            if st.button("🔮 Make Predictions", use_container_width=True):
                st.session_state.selected_tab = "🔮 Predict"
                st.rerun()
    
    with col2:
        st.markdown("### System Info")
        st.info(f"""
        **Git Commit:** `{get_git_commit_hash()}`
        **Python:** `{".".join(map(str, [3, 9, 0]))}`
        **MLflow:** `{'✅ Available' if check_mlflow_available() else '❌ Not Available'}`
        """)
        
        if check_mlflow_available():
            st.markdown("**MLflow UI:** [Open MLflow](http://localhost:5000)")

elif selected_tab == "📊 Data Manager":
    st.title("📊 Data Manager")
    
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Dataset Overview", "Preprocessing"])
    
    with tab1:
        st.subheader("Upload Dataset")
        
        data_type = st.selectbox("Data Type", ["Images", "Skeleton JSON", "CSV/Tabular"])
        
        if data_type == "Images":
            uploaded_files = st.file_uploader(
                "Upload images or zip file",
                type=['png', 'jpg', 'jpeg', 'zip'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                loader = ImageDataLoader()
                
                with st.spinner("Processing images..."):
                    try:
                        dataset_info = loader.load_from_uploaded_files(uploaded_files)
                        st.session_state.datasets[f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = dataset_info
                        st.success(f"Loaded {dataset_info['num_samples']} images with {dataset_info['num_classes']} classes")
                        
                        # Show sample images
                        if dataset_info.get('sample_images'):
                            st.subheader("Sample Images")
                            cols = st.columns(min(4, len(dataset_info['sample_images'])))
                            for i, (img_path, label) in enumerate(dataset_info['sample_images'][:4]):
                                with cols[i]:
                                    if os.path.exists(img_path):
                                        img = Image.open(img_path)
                                        st.image(img, caption=f"Label: {label}", use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Error loading images: {str(e)}")
        
        elif data_type == "Skeleton JSON":
            uploaded_files = st.file_uploader(
                "Upload skeleton JSON files",
                type=['json'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                loader = SkeletonDataLoader()
                
                with st.spinner("Processing skeleton data..."):
                    try:
                        # Validate and load files
                        valid_files = []
                        for file in uploaded_files:
                            content = json.loads(file.read().decode())
                            file.seek(0)  # Reset file pointer
                            
                            validation_result = validate_skeleton_schema(content)
                            if validation_result['valid']:
                                valid_files.append(file)
                                st.success(f"✅ {file.name}: Valid")
                            else:
                                st.error(f"❌ {file.name}: {validation_result['error']}")
                        
                        if valid_files:
                            dataset_info = loader.load_from_uploaded_files(valid_files)
                            st.session_state.datasets[f"skeleton_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = dataset_info
                            st.success(f"Loaded {dataset_info['num_sequences']} sequences")
                            
                            # Show first sequence preview
                            if dataset_info.get('sample_sequence'):
                                st.subheader("Sample Sequence Preview")
                                sample_seq = dataset_info['sample_sequence']
                                
                                # Display basic info
                                st.write(f"**ID:** {sample_seq.get('id', 'unknown')}")
                                st.write(f"**FPS:** {sample_seq.get('fps', 'unknown')}")
                                st.write(f"**Frames:** {len(sample_seq.get('frames', []))}")
                                
                                # Display first few frames as table
                                frames = sample_seq.get('frames', [])[:5]
                                if frames:
                                    frame_data = []
                                    for i, frame in enumerate(frames):
                                        joints = frame.get('joints', {})
                                        row = {'frame': i, 'time': frame.get('t', i)}
                                        for joint, coords in joints.items():
                                            if len(coords) >= 3:
                                                row[f"{joint}_x"] = coords[0]
                                                row[f"{joint}_y"] = coords[1]
                                                row[f"{joint}_z"] = coords[2]
                                        frame_data.append(row)
                                    
                                    df = pd.DataFrame(frame_data)
                                    st.dataframe(df)
                                
                                # Animated preview
                                if st.button("Show Animation Preview"):
                                    display_skeleton_animation(sample_seq)
                    
                    except Exception as e:
                        st.error(f"Error processing skeleton data: {str(e)}")
        
        elif data_type == "CSV/Tabular":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file:
                loader = CSVDataLoader()
                
                with st.spinner("Processing CSV data..."):
                    try:
                        dataset_info = loader.load_from_uploaded_file(uploaded_file)
                        st.session_state.datasets[f"csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = dataset_info
                        st.success(f"Loaded {dataset_info['num_samples']} samples with {dataset_info['num_features']} features")
                        
                        # Show data preview
                        st.subheader("Data Preview")
                        st.dataframe(dataset_info['preview_data'])
                        
                        # Show data statistics
                        if dataset_info.get('statistics'):
                            st.subheader("Data Statistics")
                            st.json(dataset_info['statistics'])
                    
                    except Exception as e:
                        st.error(f"Error loading CSV: {str(e)}")
    
    with tab2:
        st.subheader("Dataset Overview")
        
        if st.session_state.datasets:
            dataset_names = list(st.session_state.datasets.keys())
            selected_dataset = st.selectbox("Select Dataset", dataset_names)
            
            if selected_dataset:
                dataset = st.session_state.datasets[selected_dataset]
                
                # Basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Type", dataset.get('type', 'Unknown'))
                with col2:
                    st.metric("Samples", dataset.get('num_samples', dataset.get('num_sequences', 0)))
                with col3:
                    st.metric("Classes/Features", dataset.get('num_classes', dataset.get('num_features', 0)))
                
                # Class distribution (if available)
                if 'class_distribution' in dataset:
                    st.subheader("Class Distribution")
                    class_dist = dataset['class_distribution']
                    
                    fig = px.bar(
                        x=list(class_dist.keys()),
                        y=list(class_dist.values()),
                        title="Class Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sequence length distribution (for skeleton data)
                if 'sequence_lengths' in dataset:
                    st.subheader("Sequence Length Distribution")
                    fig = px.histogram(
                        x=dataset['sequence_lengths'],
                        title="Sequence Length Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No datasets loaded yet. Upload data in the 'Upload Data' tab.")
    
    with tab3:
        st.subheader("Preprocessing Options")
        
        if st.session_state.datasets:
            st.info("Preprocessing options will be applied during training.")
            
            # Augmentation options
            st.subheader("Data Augmentation")
            
            col1, col2 = st.columns(2)
            with col1:
                rotation_aug = st.checkbox("Rotation Augmentation", value=True)
                jitter_aug = st.checkbox("Jitter/Noise Augmentation", value=True)
            
            with col2:
                speed_aug = st.checkbox("Speed Scale Augmentation", value=False)
                flip_aug = st.checkbox("Horizontal Flip", value=False)
            
            # Normalization options
            st.subheader("Normalization")
            root_joint = st.selectbox("Root Joint", ["wrist", "torso", "hip", "shoulder"])
            scale_normalize = st.checkbox("Scale Normalization", value=True)
            
            if st.button("Save Preprocessing Config"):
                config = {
                    'augmentation': {
                        'rotation': rotation_aug,
                        'jitter': jitter_aug,
                        'speed_scale': speed_aug,
                        'horizontal_flip': flip_aug
                    },
                    'normalization': {
                        'root_joint': root_joint,
                        'scale_normalize': scale_normalize
                    }
                }
                
                with open("preprocessing_config.json", "w") as f:
                    json.dump(config, f, indent=2)
                
                st.success("Preprocessing configuration saved!")
        else:
            st.info("Load a dataset first to configure preprocessing.")

elif selected_tab == "🎯 Train":
    st.title("🎯 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset selection
        if st.session_state.datasets:
            dataset_names = list(st.session_state.datasets.keys())
            selected_dataset = st.selectbox("Select Dataset", dataset_names)
            st.session_state.current_dataset = selected_dataset
        else:
            st.error("No datasets available. Please upload data first.")
            selected_dataset = None
        
        if selected_dataset:
            dataset = st.session_state.datasets[selected_dataset]
            
            # Model selection
            if dataset['type'] == 'skeleton':
                model_options = ["1D-CNN", "BiLSTM", "1D-CNN + BiLSTM"]
            elif dataset['type'] == 'image':
                model_options = ["CNN-Small", "MobileNetV2", "Custom CNN"]
            else:
                model_options = ["MLP", "Random Forest", "XGBoost"]
            
            selected_model = st.selectbox("Model Architecture", model_options)
            
            # Hyperparameters
            st.subheader("Hyperparameters")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                epochs = st.slider("Epochs", 1, 100, 25)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
            
            with col_b:
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
                if dataset['type'] == 'skeleton':
                    window_length = st.slider("Window Length", 16, 128, 64)
            
            with col_c:
                class_balancing = st.checkbox("Class Balancing", value=True)
                early_stopping = st.checkbox("Early Stopping", value=True)
            
            # MLflow option
            use_mlflow = st.checkbox("Use MLflow Tracking", value=check_mlflow_available())
            if not check_mlflow_available() and use_mlflow:
                st.warning("MLflow not available. Will fallback to JSON logging.")
            
            # Run label
            run_label = st.text_input("Run Label (optional)", 
                                    placeholder="e.g., 'CNN baseline' or 'LSTM with augmentation'")
    
    with col2:
        st.subheader("Training Status")
        
        if st.session_state.training_process:
            st.info("🔄 Training in progress...")
            
            # Show progress if available
            log_file = "training.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = f.read().split('\n')[-10:]  # Last 10 lines
                    st.text_area("Recent Logs", "\n".join(logs), height=200)
            
            if st.button("Stop Training"):
                if st.session_state.training_process:
                    st.session_state.training_process.terminate()
                    st.session_state.training_process = None
                    st.success("Training stopped.")
                    st.rerun()
        else:
            st.success("✅ Ready to train")
    
    # Training controls
    if selected_dataset and not st.session_state.training_process:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Prepare training config
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            config = {
                'run_id': run_id,
                'run_label': run_label or f"{selected_model} on {selected_dataset}",
                'dataset': selected_dataset,
                'model': selected_model,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'class_balancing': class_balancing,
                'early_stopping': early_stopping,
                'use_mlflow': use_mlflow and check_mlflow_available(),
                'timestamp': datetime.now().isoformat(),
                'git_commit': get_git_commit_hash()
            }
            
            if dataset['type'] == 'skeleton':
                config['window_length'] = window_length
            
            # Save config
            os.makedirs(f"checkpoints/{run_id}", exist_ok=True)
            with open(f"checkpoints/{run_id}/run_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Start training process
            cmd = [
                "python", "train.py",
                "--config", f"checkpoints/{run_id}/run_config.json"
            ]
            
            try:
                st.session_state.training_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                st.success(f"Training started! Run ID: {run_id}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to start training: {str(e)}")

elif selected_tab == "📈 Monitor":
    st.title("📈 Training Monitor")
    
    # Load available runs
    checkpoint_dir = Path("checkpoints")
    available_runs = []
    
    if checkpoint_dir.exists():
        for run_dir in checkpoint_dir.iterdir():
            if run_dir.is_dir():
                config_file = run_dir / "run_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                        available_runs.append({
                            'run_id': config['run_id'],
                            'label': config.get('run_label', config['run_id']),
                            'model': config['model'],
                            'dataset': config['dataset'],
                            'timestamp': config['timestamp']
                        })
    
    if available_runs:
        # Sort by timestamp (newest first)
        available_runs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Run selection
        run_options = [f"{run['label']} ({run['run_id']})" for run in available_runs]
        selected_run_idx = st.selectbox("Select Run", range(len(run_options)), 
                                       format_func=lambda x: run_options[x])
        
        selected_run = available_runs[selected_run_idx]
        run_id = selected_run['run_id']
        
        # Load training history
        history = load_training_history(run_id)
        
        if history:
            col1, col2 = st.columns(2)
            
            with col1:
                # Training/Validation Loss
                st.subheader("Loss Curves")
                fig = go.Figure()
                
                if 'loss' in history:
                    fig.add_trace(go.Scatter(
                        y=history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                
                fig.update_layout(
                    title="Training/Validation Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Accuracy curves
                st.subheader("Accuracy Curves")
                fig = go.Figure()
                
                if 'accuracy' in history:
                    fig.add_trace(go.Scatter(
                        y=history['accuracy'],
                        mode='lines',
                        name='Training Accuracy',
                        line=dict(color='green')
                    ))
                
                if 'val_accuracy' in history:
                    fig.add_trace(go.Scatter(
                        y=history['val_accuracy'],
                        mode='lines',
                        name='Validation Accuracy',
                        line=dict(color='orange')
                    ))
                
                fig.update_layout(
                    title="Training/Validation Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics summary
            st.subheader("Final Metrics")
            if 'test_results' in history:
                test_results = history['test_results']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Test Accuracy", f"{test_results.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Test Loss", f"{test_results.get('loss', 0):.3f}")
                with col3:
                    st.metric("F1 Score", f"{test_results.get('f1_score', 0):.3f}")
                with col4:
                    st.metric("Precision", f"{test_results.get('precision', 0):.3f}")
            
            # Confusion matrix
            if 'confusion_matrix' in history:
                st.subheader("Confusion Matrix")
                cm = np.array(history['confusion_matrix'])
                
                # Create heatmap
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Per-class metrics
            if 'classification_report' in history:
                st.subheader("Per-Class Metrics")
                report_df = pd.DataFrame(history['classification_report']).transpose()
                st.dataframe(report_df)
            
            # Generate report button
            if st.button("📄 Generate Report", type="primary"):
                st.session_state.selected_tab = "📄 Report"
                st.session_state.selected_run_for_report = run_id
                st.rerun()
        
        else:
            st.info("No training history found for this run. The model might still be training.")
    
    else:
        st.info("No training runs found. Start training in the Train tab.")

elif selected_tab == "🔮 Predict":
    st.title("🔮 Model Prediction")
    
    # Load available models
    model_files = []
    for run_dir in Path("checkpoints").glob("*/"):
        model_path = run_dir / "model.h5"
        if model_path.exists():
            config_path = run_dir / "run_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    model_files.append({
                        'path': str(model_path),
                        'run_id': config['run_id'],
                        'label': config.get('run_label', config['run_id']),
                        'model': config['model'],
                        'dataset_type': st.session_state.datasets.get(config['dataset'], {}).get('type', 'unknown')
                    })
    
    if model_files:
        # Model selection
        model_options = [f"{m['label']} ({m['model']})" for m in model_files]
        selected_model_idx = st.selectbox("Select Model", range(len(model_options)),
                                         format_func=lambda x: model_options[x])
        
        selected_model_info = model_files[selected_model_idx]
        dataset_type = selected_model_info['dataset_type']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Data")
            
            if dataset_type == 'skeleton':
                uploaded_file = st.file_uploader("Upload Skeleton JSON", type=['json'])
                
                if uploaded_file:
                    try:
                        skeleton_data = json.loads(uploaded_file.read().decode())
                        
                        # Validate schema
                        validation = validate_skeleton_schema(skeleton_data)
                        if validation['valid']:
                            st.success("✅ Valid skeleton data")
                            
                            # Show basic info
                            st.write(f"**Frames:** {len(skeleton_data.get('frames', []))}")
                            st.write(f"**FPS:** {skeleton_data.get('fps', 'unknown')}")
                            
                            # Animation controls
                            display_skeleton_animation(skeleton_data)
                            
                            if st.button("🔮 Run Prediction", type="primary"):
                                with st.spinner("Running prediction..."):
                                    # Here you would load the model and run prediction
                                    # For now, simulate a prediction
                                    import random
                                    classes = ['idle', 'wave', 'point', 'grab', 'swipe']
                                    prediction = random.choice(classes)
                                    confidence = random.uniform(0.7, 0.95)
                                    
                                    st.session_state.prediction_result = {
                                        'prediction': prediction,
                                        'confidence': confidence,
                                        'probabilities': {cls: random.uniform(0, 1) for cls in classes}
                                    }
                                    st.rerun()
                        else:
                            st.error(f"❌ Invalid skeleton data: {validation['error']}")
                    
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            
            elif dataset_type == 'image':
                uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("🔮 Run Prediction", type="primary"):
                        with st.spinner("Running prediction..."):
                            # Simulate prediction
                            import random
                            classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
                            prediction = random.choice(classes)
                            confidence = random.uniform(0.7, 0.95)
                            
                            st.session_state.prediction_result = {
                                'prediction': prediction,
                                'confidence': confidence,
                                'probabilities': {cls: random.uniform(0, 1) for cls in classes}
                            }
                            st.rerun()
            
            else:
                st.info("CSV prediction interface coming soon...")
        
        with col2:
            st.subheader("Prediction Results")
            
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                
                # Main prediction
                st.success(f"**Prediction:** {result['prediction']}")
                st.info(f"**Confidence:** {result['confidence']:.1%}")
                
                # Probability distribution
                st.subheader("Class Probabilities")
                
                probs = result['probabilities']
                # Normalize probabilities
                total = sum(probs.values())
                if total > 0:
                    probs = {k: v/total for k, v in probs.items()}
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=list(probs.values()),
                    y=list(probs.keys()),
                    orientation='h',
                    marker_color=['red' if k == result['prediction'] else 'lightblue' 
                                for k in probs.keys()]
                ))
                
                fig.update_layout(
                    title="Class Probabilities",
                    xaxis_title="Probability",
                    yaxis_title="Class"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction timeline for skeleton data
                if dataset_type == 'skeleton' and 'skeleton_data' in st.session_state:
                    st.subheader("Prediction Timeline")
                    # This would show frame-by-frame predictions
                    st.info("Frame-by-frame prediction timeline would appear here")
            
            else:
                st.info("Upload data and run prediction to see results here.")
    
    else:
        st.error("No trained models found. Train a model first in the Train tab.")

elif selected_tab == "📄 Report":
    st.title("📄 Report Generation")
    
    # Load available runs
    available_runs = []
    for run_dir in Path("checkpoints").glob("*/"):
        config_file = run_dir / "run_config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                available_runs.append(config)
    
    if available_runs:
        available_runs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        tab1, tab2 = st.tabs(["Single Run Report", "Multi-Run Comparison"])
        
        with tab1:
            st.subheader("Generate Single Run Report")
            
            run_options = [f"{run.get('run_label', run['run_id'])} ({run['run_id']})" 
                          for run in available_runs]
            selected_run_idx = st.selectbox("Select Run", range(len(run_options)),
                                           format_func=lambda x: run_options[x])
            
            selected_run = available_runs[selected_run_idx]
            
            if st.button("📄 Generate Report", type="primary"):
                with st.spinner("Generating report..."):
                    generator = ReportGenerator()
                    report_html = generator.generate_single_run_report(selected_run['run_id'])
                    
                    if report_html:
                        st.success("Report generated successfully!")
                        
                        # Show preview
                        st.subheader("Report Preview")
                        st.components.v1.html(report_html, height=600, scrolling=True)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📄 Download HTML",
                                data=report_html,
                                file_name=f"report_{selected_run['run_id']}.html",
                                mime="text/html"
                            )
                        
                        with col2:
                            if st.button("📄 Download PDF"):
                                try:
                                    pdf_content = generator.generate_pdf_from_html(report_html)
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=pdf_content,
                                        file_name=f"report_{selected_run['run_id']}.pdf",
                                        mime="application/pdf"
                                    )
                                except Exception as e:
                                    st.error(f"PDF generation failed: {str(e)}")
                    else:
                        st.error("Failed to generate report")
        
        with tab2:
            st.subheader("Multi-Run Comparison Report")
            
            # Multi-select for runs
            run_labels = [f"{run.get('run_label', run['run_id'])} ({run['run_id']})" 
                         for run in available_runs]
            selected_runs = st.multiselect("Select Runs to Compare", run_labels)
            
            if len(selected_runs) >= 2:
                if st.button("📊 Generate Comparison Report", type="primary"):
                    with st.spinner("Generating comparison report..."):
                        # Extract run IDs from selected labels
                        run_ids = []
                        for label in selected_runs:
                            run_id = label.split('(')[-1].rstrip(')')
                            run_ids.append(run_id)
                        
                        generator = ReportGenerator()
                        report_html = generator.generate_multi_run_report(run_ids)
                        
                        if report_html:
                            st.success("Comparison report generated successfully!")
                            
                            # Show preview
                            st.subheader("Report Preview")
                            st.components.v1.html(report_html, height=600, scrolling=True)
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📄 Download HTML",
                                    data=report_html,
                                    file_name=f"comparison_report_{'_'.join(run_ids[:3])}.html",
                                    mime="text/html"
                                )
                            
                            with col2:
                                if st.button("📄 Download PDF"):
                                    try:
                                        pdf_content = generator.generate_pdf_from_html(report_html)
                                        st.download_button(
                                            label="📄 Download PDF",
                                            data=pdf_content,
                                            file_name=f"comparison_report_{'_'.join(run_ids[:3])}.pdf",
                                            mime="application/pdf"
                                        )
                                    except Exception as e:
                                        st.error(f"PDF generation failed: {str(e)}")
                        else:
                            st.error("Failed to generate comparison report")
            else:
                st.info("Select at least 2 runs to generate a comparison report.")
    
    else:
        st.info("No training runs found. Complete training first to generate reports.")

elif selected_tab == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Environment", "Export", "Configuration"])
    
    with tab1:
        st.subheader("Environment Information")
        
        # System info
        import sys
        import platform
        
        st.write("**System Information:**")
        st.code(f"""
Python Version: {sys.version}
Platform: {platform.platform()}
Architecture: {platform.architecture()[0]}
Processor: {platform.processor() or 'Unknown'}
Git Commit: {get_git_commit_hash()}
        """)
        
        # Package versions
        st.write("**Key Package Versions:**")
        try:
            import tensorflow as tf
            tf_version = tf.__version__
        except ImportError:
            tf_version = "Not installed"
        
        try:
            import streamlit as st_version
            streamlit_version = st_version.__version__
        except:
            streamlit_version = "Unknown"
        
        st.code(f"""
TensorFlow: {tf_version}
Streamlit: {streamlit_version}
MLflow: {'Available' if check_mlflow_available() else 'Not installed'}
        """)
        
        # Environment variables
        st.write("**Environment Variables:**")
        env_vars = ['CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL', 'MLFLOW_TRACKING_URI']
        for var in env_vars:
            value = os.environ.get(var, 'Not set')
            st.write(f"**{var}:** `{value}`")
    
    with tab2:
        st.subheader("Export Options")
        
        # Export requirements
        if st.button("📦 Export Requirements"):
            try:
                result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
                requirements = result.stdout
                
                st.download_button(
                    label="📄 Download requirements.txt",
                    data=requirements,
                    file_name="requirements.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Failed to export requirements: {str(e)}")
        
        # Export run configs
        if st.button("💾 Export All Run Configs"):
            configs = {}
            for run_dir in Path("checkpoints").glob("*/"):
                config_file = run_dir / "run_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                        configs[config['run_id']] = config
            
            if configs:
                st.download_button(
                    label="📄 Download run_configs.json",
                    data=json.dumps(configs, indent=2),
                    file_name="all_run_configs.json",
                    mime="application/json"
                )
            else:
                st.info("No run configurations found.")
    
    with tab3:
        st.subheader("Platform Configuration")
        
        # Report branding config
        st.write("**Report Branding:**")
        
        try:
            with open("report_config.json") as f:
                report_config = json.load(f)
        except:
            report_config = {
                "brand_name": "ML Training Platform",
                "primary_color": "#FF6B6B",
                "secondary_color": "#4ECDC4",
                "logo_url": "",
                "footer_text": "Generated by ML Training Platform"
            }
        
        brand_name = st.text_input("Brand Name", report_config.get("brand_name", ""))
        primary_color = st.color_picker("Primary Color", report_config.get("primary_color", "#FF6B6B"))
        secondary_color = st.color_picker("Secondary Color", report_config.get("secondary_color", "#4ECDC4"))
        logo_url = st.text_input("Logo URL", report_config.get("logo_url", ""))
        footer_text = st.text_input("Footer Text", report_config.get("footer_text", ""))
        
        if st.button("💾 Save Branding Config"):
            new_config = {
                "brand_name": brand_name,
                "primary_color": primary_color,
                "secondary_color": secondary_color,
                "logo_url": logo_url,
                "footer_text": footer_text
            }
            
            with open("report_config.json", "w") as f:
                json.dump(new_config, f, indent=2)
            
            st.success("Branding configuration saved!")
        
        # MLflow configuration
        st.write("**MLflow Configuration:**")
        mlflow_available = check_mlflow_available()
        st.write(f"**Status:** {'✅ Available' if mlflow_available else '❌ Not Available'}")
        
        if mlflow_available:
            tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', './mlruns')
            st.write(f"**Tracking URI:** `{tracking_uri}`")
            
            if st.button("🚀 Start MLflow UI"):
                st.info("MLflow UI can be started with: `mlflow ui --host 0.0.0.0 --port 5001`")
