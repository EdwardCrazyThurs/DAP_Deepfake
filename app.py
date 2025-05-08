from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_executor import Executor
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import io
import requests
import tempfile
import atexit
import shutil
import torch
import logging
from werkzeug.utils import secure_filename
from functools import wraps


# Initialize Flask app
app = Flask(__name__)
CORS(app)
executor = Executor(app)

# Configure rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'mp4', 'wav', 'mp3', 'jpg', 'jpeg', 'png'}

# Create temp directory for file processing
temp_dir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

# Lazy-loaded models
models_loaded = {
    'audio': None,
    'image': None
}

class TimeoutException(Exception):
    pass

def get_audio_model():
    if models_loaded['audio'] is None:
        from tensorflow.keras.models import load_model
        models_loaded['audio'] = load_model('utils/model_audio_fakedetection.h5')
    return models_loaded['audio']

def get_image_model():
    if models_loaded['image'] is None:
        
        from utils.image_model import DeepfakeDetectionModel

        logger = logging.getLogger(__name__)
        model = DeepfakeDetectionModel()

        try:
            state_dict = torch.load('utils/deepfake_efficientnet_pytorch.pth', 
                                    map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Model weight loading failed: {str(e)}")
            raise e

        model.eval()
        models_loaded['image'] = model

    return models_loaded['image']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_s3(file_data, file_type):
    """Placeholder for actual S3 upload implementation"""
    return f"https://your-storage-service.com/{file_type}_{secure_filename(file_data.filename)}"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/generate')
def generate_page():
    return render_template('generate.html')

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    try:
        if 'media' not in request.files:
            return jsonify({'error': 'Missing media file'}), 400

        media_file = request.files['media']
        if not media_file or not allowed_file(media_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        file_ext = media_file.filename.rsplit('.', 1)[1].lower()
        
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=f'.{file_ext}', delete=True) as temp_file:
            media_file.save(temp_file.name)
            
            # Handle audio files
            if file_ext in ['mp4', 'wav', 'mp3']:
                from utils.audio_preprocess import process_uploaded_audio
                mfcc, f0, pitch_var = process_uploaded_audio(temp_file.name)
                model = get_audio_model()
                y_pred = model.predict([mfcc, f0, pitch_var])
                return jsonify({'confidence': round(float(y_pred.flatten()[0]), 4)}), 200
            
            # Handle image files
            elif file_ext in ['jpg', 'jpeg', 'png']:
                from PIL import Image
                import torch
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img = Image.open(temp_file.name).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    model = get_image_model()
                    output = model(img_tensor)
                    softmax_output = torch.softmax(output, dim=1)[0]

                    confidence_fake = softmax_output[0].item()
                    confidence_real = softmax_output[1].item()
                    
                    # Define threshold – e.g., only mark as fake if fake confidence > 0.9
                    if confidence_fake > confidence_real:
                        prediction_label = 'fake'
                        confidence = confidence_fake
                    else:
                        prediction_label = 'real'
                        confidence = confidence_real
                        
                    return jsonify({
                        'prediction': prediction_label,
                        'confidence': round(confidence, 4)
                    }), 200
            
            else:
                return jsonify({'error': 'Unsupported file type'}), 400

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
@limiter.limit("5 per minute")
def generate_lipsync():
    if 'video' not in request.files or 'api_key' not in request.form:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    future = executor.submit(process_generation, request)
    return jsonify({'job_id': future.job_id}), 202

def process_generation(request):
    try:
        video_file = request.files['video']
        api_key = request.form['api_key']
        model = request.form.get('model', 'lipsync-2')
        sync_mode = request.form.get('sync_mode', 'bounce')
        
        video_url = upload_to_s3(video_file, 'video')
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        
        input_data = [{"type": "video", "url": video_url}]

        if 'text' in request.form:
            return {'error': 'Text input is not yet implemented'}
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            audio_url = upload_to_s3(audio_file, 'audio')
            input_data.append({"type": "audio", "url": audio_url})
        else:
            return {'error': 'Missing audio or text input'}

        data = {
            "model": model,
            "input": input_data,
            "options": {
                "sync_mode": sync_mode,
                "active_speaker": True,
                "output_format": "mp4"
            },
            "webhookUrl": request.host_url.rstrip('/') + '/api/webhook'
        }

        response = requests.post(
            'https://api.sync.so/v2/generate',
            headers=headers,
            json=data
        )

        if response.status_code in (200, 201, 202):
            return response.json()
        else:
            return {'error': f"API request failed: {response.text}"}

    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return {'error': str(e)}

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    try:
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400

        response = requests.get(
            f'https://api.sync.so/v2/generate/{job_id}',
            headers={'x-api-key': api_key}
        )

        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({'error': f"Status check failed: {response.text}"}), response.status_code

    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/webhook', methods=['POST'])
def webhook_handler():
    try:
        payload = request.json

        if payload['status'] == 'COMPLETED':
            app.config[f"result_{payload['id']}"] = payload['outputUrl']
            return jsonify({'status': 'success'}), 200
        
        elif payload['status'] == 'FAILED':
            logger.error(f"Generation failed: {payload.get('error')}")
            return jsonify({'status': 'failed', 'error': payload.get('error')}), 200
        
        return jsonify({'status': 'processing'}), 200

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    try:
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400

        result_url = app.config.get(f"result_{job_id}")
        if not result_url:
            return jsonify({'error': 'Result not found'}), 404

        response = requests.get(
            result_url,
            headers={'x-api-key': api_key},
            stream=True
        )

        if response.status_code == 200:
            return send_file(
                io.BytesIO(response.content),
                mimetype='video/mp4',
                as_attachment=True,
                download_name=f'lipsync_{job_id}.mp4'
            )
        else:
            return jsonify({'error': 'Failed to download result'}), response.status_code

    except Exception as e:
        logger.error(f"Result download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
