from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import io
import requests
import json
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.keras.models import load_model
from utils.audio_preprocess import process_uploaded_audio 

app = Flask(__name__)
CORS(app)

# Configure upload folder for temporary file storage
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'wav', 'mp3', 'jpg', 'jpeg', 'png'}

# load audio deepfake detection model
model = load_model('utils/model_audio_fakedetection.h5')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_s3(file_data, file_type):
    """
    In a real application, you would upload the file to S3 or another storage service
    and return the URL. For this demo, we'll assume this step and return a placeholder.
    """
    # TODO: Implement actual file upload to S3 or similar service
    return f"https://your-storage-service.com/{file_type}_{secure_filename(file_data.filename)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/generate')
def generate_page():
    return render_template('generate.html')

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    try:
        if 'media' not in request.files:
            return jsonify({'error': 'Missing media file'}), 400

        media_file = request.files['media']
        mode = request.form.get('mode', 'comprehensive')
        threshold = float(request.form.get('threshold', '0.5'))

        if not media_file or not allowed_file(media_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # TODO: Implement actual deepfake detection
        # if audio:
        if media_file.filename.rsplit('.', 1)[1].lower() in ['mp4', 'wav', 'mp3']:
            mfcc, f0, pitch_var = process_uploaded_audio(media_file)
            # inference
            y_pred = model.predict([mfcc, f0, pitch_var])
            confidence = float(y_pred.flatten()[0])
            return jsonify({
                'confidence': round(confidence, 4)
            }), 200

        else:  # for image
            import torch
            from torchvision import transforms
            from PIL import Image
            from utils.models.image_model import DeepfakeDetectionModel

            # Load model once (global caching optional)
            image_model = DeepfakeDetectionModel()
            image_model.load_state_dict(torch.load('utils/deepfake_efficientnet_pytorch.pth', map_location=torch.device('cpu')))
            image_model.eval()

            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            # Read image file from memory
            img = Image.open(media_file.stream).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            # Inference
            with torch.no_grad():
                output = image_model(img_tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = float(torch.max(output).item())

            return jsonify({
                'prediction': 'real' if prediction == 1 else 'fake',
                'confidence': round(confidence, 4)
            }), 200


    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_lipsync():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'Missing video file'}), 400

        if 'api_key' not in request.form:
            return jsonify({'error': 'Missing API key'}), 400

        video_file = request.files['video']
        api_key = request.form['api_key']
        model = request.form.get('model', 'lipsync-2')
        sync_mode = request.form.get('sync_mode', 'bounce')
        
        # Upload video to storage (placeholder)
        video_url = upload_to_s3(video_file, 'video')
        
        # Prepare API request
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        
        input_data = [
            {
                "type": "video",
                "url": video_url
            }
        ]

        # Handle text or audio input
        if 'text' in request.form:
            # TODO: Implement text-to-speech conversion and audio upload
            # For now, we'll return an error as text input requires additional processing
            return jsonify({'error': 'Text input is not yet implemented'}), 501
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            audio_url = upload_to_s3(audio_file, 'audio')
            input_data.append({
                "type": "audio",
                "url": audio_url
            })
        else:
            return jsonify({'error': 'Missing audio or text input'}), 400

        # Prepare request body
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

        # Make request to Sync API
        response = requests.post(
            'https://api.sync.so/v2/generate',
            headers=headers,
            json=data
        )

        if response.status_code in (200, 201, 202):
            return jsonify(response.json()), response.status_code
        else:
            return jsonify({'error': f"API request failed: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        return jsonify({'error': str(e)}), 500

@app.route('/api/webhook', methods=['POST'])
def webhook_handler():
    try:
        payload = request.json

        if payload['status'] == 'COMPLETED':
            # Store the result URL
            app.config[f"result_{payload['id']}"] = payload['outputUrl']
            return jsonify({'status': 'success'}), 200
        
        elif payload['status'] == 'FAILED':
            print(f"Generation failed: {payload.get('error')}")
            return jsonify({'status': 'failed', 'error': payload.get('error')}), 200
        
        return jsonify({'status': 'processing'}), 200

    except Exception as e:
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

        # Download the result from Sync
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
