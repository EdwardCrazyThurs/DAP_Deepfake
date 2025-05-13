import numpy as np
import librosa
from pydub import AudioSegment
import io
from sklearn.preprocessing import StandardScaler


# scale the features: mfcc
def safe_scale(features):
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    scaler = StandardScaler()
    return scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)

# scale the features: F0
def process_f0(f0_array):
    f0_array = np.where(f0_array < 1, np.nan, f0_array)  # remove invalid: <1Hz
    f0_mean = np.nanmean(f0_array, axis=1, keepdims=True)
    f0_std = np.nanstd(f0_array, axis=1, keepdims=True) + 1e-6
    return np.nan_to_num((f0_array - f0_mean) / f0_std, nan=0.0)

# process audio: load + extract features + scale
def process_uploaded_audio(file, target_duration=2.0, sr=16000, silence_threshold=1e-4, n_mfcc=40):
    # load audio
    #filename = file.filename.lower()
    if file.endswith('.mp4') : # or filename.endswith('.mov') or filename.endswith('.avi')
        # if video: extract audio & read
        video = AudioSegment.from_file(file, format="mp4")
        buffer = io.BytesIO()
        video.export(buffer, format="wav")
        buffer.seek(0)
        file_bytes = buffer.read()
    elif file.endswith('.wav') or file.endswith('.mp3'):
        # if audio: read
        with open(file, 'rb') as f:
            file_bytes = f.read() 
    y, _ = librosa.load(io.BytesIO(file_bytes), sr=sr)

    # remove silent audio
    if np.mean(np.abs(y)) < silence_threshold:
        return None
    # ensure 2 seconds
    target_samples = int(target_duration * sr)
    if len(y) > target_samples:
        y = y[:target_samples]
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')

    # feature extraction: MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # feature extraction: F0
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    f0 = np.nan_to_num(f0)
    # feature extraction: pitch variation
    pitch_variation = np.diff(f0)

    # scale
    X_mfcc = np.array([mfcc])
    X_f0 = np.array([f0])
    X_pitch_var = np.array([pitch_variation])
    X_mfcc = safe_scale(X_mfcc)
    X_f0 = process_f0(X_f0)
    X_pitch_var = np.clip(X_pitch_var, -1e4, 1e4)  # remove extreme value
    X_pitch_var = StandardScaler().fit_transform(X_pitch_var)

    return  X_mfcc, X_f0, X_pitch_var