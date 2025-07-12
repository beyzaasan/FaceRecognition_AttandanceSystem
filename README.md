# Biometric Access System

A sophisticated face recognition-based access control system built with Streamlit, MySQL, and advanced computer vision technologies.

## 🚀 Features

- **Advanced Face Recognition**: Multi-stage pipeline with MTCNN detection, dlib alignment, and custom recognition
- **Real-time Access Control**: Live camera feed with instant face recognition
- **Comprehensive Person Management**: Add, edit, and manage registered persons
- **Detailed Access Logs**: Track all entry/exit events with confidence scores
- **Advanced Attendance Reports**: Generate detailed reports with working hours analysis
- **Multi-format Export**: Export data to CSV and Excel with color coding
- **Camera Location Support**: Multiple camera locations (entry, exit, internal doors)

## 🔧 Technology Stack

### Core Technologies
- **Streamlit** - Web application framework
- **MySQL** - Database management
- **OpenCV** - Computer vision operations

### Face Recognition Pipeline
- **MTCNN** - Multi-task Cascaded Convolutional Networks for face detection
- **dlib** - Facial landmarks detection and face alignment
- **Custom FaceRecognizer** - Advanced embedding extraction and matching
- **Face Alignment** - Automatic face normalization for better recognition

### Machine Learning
- **dlib face_recognition_model_v1** - Primary embedding extraction
- **scikit-learn** - Similarity calculations and analysis

## Quick Start

### 🚀 **One-Command Setup & Run**

```bash
# Clone the repository
git clone <repository-url>
cd FaceRecognition_AttandanceSystem

# Download required model files (IMPORTANT!)
python download_models.py

# Run the application (automatically handles setup)
python run.py
```

The script will:
- ✅ Check all requirements
- ✅ Create `.env` file from template if needed
- ✅ Install dependencies if missing
- ✅ Start the application

### 📦 **Manual Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FaceRecognition_AttandanceSystem
   ```

2. **Download AI Model Files** ⚠️ **REQUIRED**
   ```bash
   # Download all required model files (~130MB total)
   python download_models.py
   ```
   
   **Why download models?**
   - Model files are large (~130MB) and not included in the repository
   - These files are required for face detection and recognition
   - The download script will automatically fetch all necessary models

3. **Install dependencies**
   ```bash
   python run.py --install
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env file with your database credentials
   nano .env
   ```

5. **Configure database**
   - Create a MySQL database named `biometric_system`
   - Import the database schema from `database/biometric_system.sql`
   ```bash
   mysql -u root -p biometric_system < database/biometric_system.sql
   ```

6. **Run the application**
   ```bash
   python run.py
   ```

## 📥 Required Model Files

The application requires several AI model files for face recognition functionality. These files are **not included** in the repository due to their large size (~130MB total).

### Automatic Download (Recommended)
```bash
python download_models.py
```

### Manual Download
If automatic download fails, you can manually download the models:

1. **dlib Models** (from http://dlib.net/files/):
   - `dlib_face_recognition_resnet_model_v1.dat.bz2` → Extract to `models/dlib_face_recognition_resnet_model_v1.dat`
   - `shape_predictor_68_face_landmarks.dat.bz2` → Extract to `models/shape_predictor_68_face_landmarks.dat`

2. **OpenCV Models** (from OpenCV repository):
   - `res10_300x300_ssd_iter_140000.caffemodel` → Save to `models/res10_300x300_ssd_iter_140000.caffemodel`
   - `deploy.prototxt` → Save to `models/deploy.prototxt`
   - `opencv_face_detector.pbtxt` → Save to `models/opencv_face_detector.pbtxt`
   - `opencv_face_detector_uint8.pb` → Save to `models/opencv_face_detector_uint8.pb`

### Model File Sizes
- `dlib_face_recognition_resnet_model_v1.dat`: ~95MB
- `shape_predictor_68_face_landmarks.dat`: ~68MB
- `res10_300x300_ssd_iter_140000.caffemodel`: ~10MB
- `opencv_face_detector_uint8.pb`: ~2.6MB
- `deploy.prototxt`: ~3.5KB
- `opencv_face_detector.pbtxt`: ~34KB

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
DB_HOST=localhost
DB_NAME=biometric_system
DB_USER=root
DB_PASSWORD=your_password_here
DB_CHARSET=utf8mb4
DB_COLLATION=utf8mb4_unicode_ci
```

## Usage

### 🎯 **Running the Application**

```bash
# Normal run
python run.py

# Install dependencies only
python run.py --install
```

### 📱 **Application Features**

1. **👥 Add Person**: 
   - Upload photos or capture from camera
   - Advanced face detection with MTCNN
   - Automatic face alignment and embedding extraction
   - Multiple embedding methods for reliability

2. **📋 View Persons**: 
   - Manage registered persons and their status
   - View face embeddings and registration details
   - Activate/deactivate persons
   - Delete persons with confirmation

3. **📊 Access Logs**: 
   - Real-time access monitoring
   - Filter by person, location, and access type
   - Confidence scores for each recognition
   - Export to CSV/Excel with color coding

4. **📈 Attendance Reports**: 
   - Individual employee reports
   - Monthly summary reports
   - Department-wise analysis
   - Working hours calculation and analysis

5. **🔍 Face Recognition Test**: 
   - Real-time camera testing
   - Multiple camera location support
   - Instant recognition feedback
   - Access type simulation (entry/exit/internal)

## Security Notes

- Never commit the `.env` file to version control
- Keep your database credentials secure
- The `.env` file is already added to `.gitignore`

## 📋 Requirements

### System Requirements
- **Python 3.7+**
- **MySQL 5.7+**
- **Webcam** for face recognition testing
- **4GB+ RAM** (for MTCNN and dlib models)
- **Internet connection** for initial model download

### Key Dependencies
- **streamlit** - Web application
- **mysql-connector-python** - Database connectivity
- **opencv-python** - Computer vision
- **mtcnn** - Face detection
- **dlib** - Facial landmarks and recognition
- **scikit-learn** - Machine learning utilities
- **streamlit-webrtc** - Real-time camera streaming

## 🛠️ Troubleshooting

### Common Issues

1. **Missing model files**
   ```bash
   # Download all required models
   python download_models.py
   ```

2. **Missing dependencies**
   ```bash
   python run.py --install
   ```

3. **Environment file not found**
   - The script will automatically create `.env` from `.env.example`
   - Edit the `.env` file with your database credentials

4. **Database connection issues**
   - Check your MySQL server is running
   - Verify credentials in `.env` file
   - Ensure database `biometric_system` exists

5. **Face recognition issues**
   - Ensure webcam is working and accessible
   - Check lighting conditions for better recognition
   - Verify dlib models are downloaded correctly
   - MTCNN may require more RAM (4GB+ recommended)

6. **Model download issues**
   ```bash
   # Try manual download if automatic fails
   python download_models.py
   ```

### Script Features

- 🔍 **Automatic checks**: Validates all required files and dependencies
- 📦 **Smart installation**: Only installs missing packages
- 🛡️ **Error handling**: Clear error messages and solutions
- 🚀 **One-command setup**: Handles everything automatically

## 🏗️ Architecture

### Face Recognition Pipeline
```
Input Image → MTCNN Detection → dlib Landmarks → Face Alignment → Embedding Extraction → Recognition
```

### Database Schema
- **persons**: Person registration and face embeddings
- **access_logs**: Real-time access tracking
- **attendance_analysis**: Working hours and attendance data

### Camera Locations
- **Main Entry**: Primary building entrance
- **Main Exit**: Primary building exit  
- **Internal Door 1**: Internal access control
- **Internal Door 2**: Additional internal control

## 🔄 Recent Updates

### v2.0 - Advanced Face Recognition Pipeline
- ✅ **MTCNN Integration**: Replaced basic face detection with MTCNN for higher accuracy
- ✅ **Custom Face Processing Classes**: Implemented modular face detection, alignment, and recognition
- ✅ **dlib Face Recognition**: Added dlib-based embedding extraction as fallback method
- ✅ **Face Alignment**: Automatic face normalization using 68-point landmarks
- ✅ **Multiple Embedding Methods**: 6 different approaches for reliable embedding extraction
- ✅ **Enhanced Error Handling**: Better debugging and error recovery
- ✅ **Optimized Performance**: Improved real-time processing capabilities

### Key Improvements
- **Detection Accuracy**: MTCNN provides superior face detection
- **Recognition Reliability**: Multiple embedding methods ensure successful extraction
- **Face Quality**: Automatic alignment improves recognition accuracy
- **System Stability**: Better error handling and fallback mechanisms

## 📁 Project Structure

```
FaceRecognition_AttandanceSystem/
├── app/
│   ├── __init__.py
│   └── biometric_app.py          # Main Streamlit application
├── database/
│   └── biometric_system.sql      # Database schema
├── models/                       # AI/ML model files (downloaded separately)
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── deploy.prototxt
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── download_models.py            # Model download script
├── requirements.txt              # Python dependencies
├── run.py                       # Application entry point
├── LICENSE                       # MIT License
└── README.md                    # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

- **Model Files**: The models directory contains large AI model files (~130MB total). These are required for face recognition functionality and must be downloaded separately using `python download_models.py`.
- **Database**: Make sure to set up MySQL and import the schema before running the application.
- **Environment**: Always use the `.env.example` as a template and never commit your actual `.env` file.
- **Webcam**: The application requires a working webcam for face recognition testing.
- **Internet**: Initial setup requires internet connection to download model files.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your database configuration
4. Check that your webcam is accessible
5. Make sure model files are downloaded (`python download_models.py`)
6. Open an issue on GitHub with detailed error information
