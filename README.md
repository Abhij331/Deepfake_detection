# Deepfake Detection Model (Video & Image)

A robust deep learning system for detecting deepfake content in both videos and images using EfficientNetB0 and custom neural networks.

## Features

- Real-time deepfake detection for videos and images
- Support for multiple file formats (mp4, avi, mov, jpg, jpeg, png)
- Frame-by-frame analysis with visual feedback
- RESTful API interface
- Confidence scoring system
- Anomaly detection and reporting

## Tech Stack

- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow, EfficientNetB0
- **Image Processing**: OpenCV
- **Frontend Integration**: CORS support for frontend applications

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- CUDA (for GPU support)
- At least 4GB RAM
- 100MB+ free disk space

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd deepfake-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model:
- Place your trained model file as `temp_model.h5` in the root directory

## Project Structure

```
├── backend/
│   ├── app.py            # Main Flask application
│   ├── uploads/          # Temporary storage for uploaded files
│   └── static/          # Storage for processed frames
├── requirements.txt      # Python dependencies
└── temp_model.h5        # Trained model file
```

## API Endpoints

### 1. Health Check
- **URL**: `/`
- **Method**: `GET`
- **Response**: Confirms if API is running

### 2. Prediction
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: Form-data with file upload
- **Supported Files**: Videos (mp4, avi, mov) or Images (jpg, jpeg, png)
- **Max File Size**: 100MB
- **Response Format**:
```json
{
    "status": "success",
    "prediction": "Real/Fake",
    "confidence": float,
    "frames": [string],
    "anomalies": [string]
}
```

### 3. Static Files
- **URL**: `/static/<filename>`
- **Method**: `GET`
- **Response**: Processed frame images

## Configuration

- Server runs on `http://localhost:5000`
- CORS enabled for frontend origin `http://localhost:3000`
- Maximum file size: 100MB
- Automatic cleanup of processed files after 1 hour

## Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- Oversized files
- Missing files
- Processing errors

## Security Features

- Secure filename handling
- Automatic file cleanup
- CORS protection
- Temporary file management

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EfficientNet model architecture
- TensorFlow team
- OpenCV contributors 