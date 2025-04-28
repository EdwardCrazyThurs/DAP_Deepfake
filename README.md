# Deepfake Detection and Generation Web Application

This web application provides two main functionalities:
1. Deepfake Detection: Upload images or audio files to detect if they are deepfakes
2. Deepfake Generation: Generate talking face videos from an image and text input

## Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── static/
│   ├── css/
│   │   └── style.css     # Application styles
│   └── js/
│       └── main.js       # Client-side JavaScript
├── templates/
│   └── index.html        # Main HTML template
└── uploads/              # Directory for uploaded files
```

## Deployment to Render

1. Create a new Web Service on Render
2. Connect your repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

## API Endpoints

### POST /detect
Endpoint for deepfake detection
- Input: Form data with 'file' field (image or audio)
- Output: JSON with detection results

### POST /generate
Endpoint for deepfake generation
- Input: Form data with 'image' field and 'text' field
- Output: JSON with generation status

## Notes

- The ML models for detection and generation are not implemented in this framework
- The upload directory is created automatically when the application starts
- File uploads are limited to specific extensions (png, jpg, jpeg, gif, wav, mp3)
- The application includes proper error handling and user feedback
- The UI is responsive and works on both desktop and mobile devices 