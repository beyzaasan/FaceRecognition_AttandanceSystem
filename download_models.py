#!/usr/bin/env python3
"""
Model Download Script for Face Recognition System

This script downloads the required model files for the face recognition system.
The models are large files (~130MB total) and are not included in the repository.
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path

# Model URLs and file information
MODELS = {
    "dlib_face_recognition_resnet_model_v1.dat": {
        "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        "size": "95MB",
        "description": "dlib face recognition model"
    },
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "size": "68MB",
        "description": "68-point facial landmarks predictor"
    },
    "res10_300x300_ssd_iter_140000.caffemodel": {
        "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "size": "10MB",
        "description": "OpenCV face detection model"
    },
    "deploy.prototxt": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "size": "3.5KB",
        "description": "OpenCV face detection configuration"
    },
    "opencv_face_detector.pbtxt": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
        "size": "34KB",
        "description": "OpenCV face detector configuration"
    },
    "opencv_face_detector_uint8.pb": {
        "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
        "size": "2.6MB",
        "description": "OpenCV face detector model"
    }
}

def download_file(url, filename, models_dir):
    """Download a file from URL to the models directory."""
    filepath = models_dir / filename
    
    if filepath.exists():
        print(f"✅ {filename} already exists, skipping...")
        return True
    
    print(f"📥 Downloading {filename}...")
    print(f"   URL: {url}")
    print(f"   Size: {MODELS[filename]['size']}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def extract_bz2(filename, models_dir):
    """Extract .bz2 files (for dlib models)."""
    filepath = models_dir / filename
    if not filepath.exists():
        return False
    
    if filename.endswith('.bz2'):
        import bz2
        extracted_name = filename[:-4]  # Remove .bz2
        extracted_path = models_dir / extracted_name
        
        if extracted_path.exists():
            print(f"✅ {extracted_name} already extracted, skipping...")
            return True
        
        print(f"📦 Extracting {filename}...")
        try:
            with bz2.open(filepath, 'rb') as source, open(extracted_path, 'wb') as target:
                target.write(source.read())
            print(f"✅ Successfully extracted {extracted_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to extract {filename}: {e}")
            return False
    
    return True

def main():
    """Main function to download all required models."""
    print("🚀 Face Recognition Model Downloader")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"📁 Models will be downloaded to: {models_dir.absolute()}")
    print()
    
    # Download all models
    success_count = 0
    total_count = len(MODELS)
    
    for filename, info in MODELS.items():
        print(f"📋 {info['description']}")
        if download_file(info['url'], filename, models_dir):
            if extract_bz2(filename, models_dir):
                success_count += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Download Summary:")
    print(f"   ✅ Successfully downloaded: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("💡 You can now run the face recognition application.")
    else:
        print("⚠️  Some models failed to download.")
        print("💡 Please check your internet connection and try again.")
        print("💡 You can also download models manually from the URLs above.")

if __name__ == "__main__":
    main() 