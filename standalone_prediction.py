import os
import sys
import argparse
import subprocess
import shutil
import librosa
import soundfile as sf
import numpy as np
import torch
import warnings
from scipy.signal import butter, filtfilt
from joblib import load


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    print("ERROR, You must install speechBrain library")
    sys.exit(1)

#SETTINGS
INPUT_MP3_FOLDER = 'input_mp3'
TEMP_FOLDER = 'temp_files'

MODEL_PATH = 'trained_models/full_trained/xgboost_xvector_model.joblib'
SCALER_PATH = 'trained_models/full_trained/xgboost_xvector_scaler.joblib'
ENCODER_PATH = 'trained_models/full_trained/xgboost_xvector_classes.joblib'
#SPEECHBRAIN_MODEL_CACHE = 'pretrained_models/spkrec-xvect-voxceleb'


#PARAMETERS
MAX_DURATION_SECONDS = 10 #Maximum length of mp3 file
TARGET_FREQUENCY = 16000
VAD_THRESHOLD_DB = 25
APPLY_FILTER = True
FILTER_CUTOFF_HZ = 80.0
FILTER_ORDER = 5

#SCRIPT

def load_speechbrain_model():
    try:
        print("Loading the SpeechBrain model")
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            #savedir=SPEECHBRAIN_MODEL_CACHE
        )
        print("SpeechBrain model loaded")
        return model
    except Exception as e:
        print(f"Error with SpeechBrain model {e}")
        return None

def check_duration(file_path):
    try:
        duration = librosa.get_duration(path=file_path)
        if duration > MAX_DURATION_SECONDS:
            print(f"Saved audio is longer than {MAX_DURATION_SECONDS} seconds ({duration:.2f}s)")
            return False
        print(f"Audio length: {duration:.2f}s (OK)")
        return True
    except Exception as e:
        print(f"Error with decoding input file: {e}")
        return False

def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        cmd = ['ffmpeg', '-i', mp3_path, '-hide_banner', '-loglevel', 'error', '-y', wav_path]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"File succesfully converted to wav: {os.path.basename(wav_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error with FFmpeg: {e.stderr.decode('utf-8', errors='ignore')}")
        return False
    except FileNotFoundError:
        print("Ffmpeg not found or not in system path")
        return False

def preprocess_audio(input_path, output_path):
    try:
        y, sr_original = librosa.load(input_path, sr=None)
        if sr_original != TARGET_FREQUENCY:
            y_resampled = librosa.resample(y, orig_sr=sr_original, target_sr=TARGET_FREQUENCY)
        else:
            y_resampled = y
        sr_new = TARGET_FREQUENCY
        y_normalized = librosa.util.normalize(y_resampled)
        y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=VAD_THRESHOLD_DB) 
        y_final = y_trimmed if len(y_trimmed) > 0 else y_normalized
        if APPLY_FILTER and len(y_final) > 0:
            b, a = butter(FILTER_ORDER, FILTER_CUTOFF_HZ / (0.5 * sr_new), btype='high', analog=False)
            y_final = filtfilt(b, a, y_final)
        sf.write(output_path, y_final, sr_new)
        print(f"Audio preprocessing done and saved in: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"Error with preprocessing: {e}")
        return False


def extract_x_vector(processed_wav_path, spk_model):
    try:
        print("Extracting X-Vector")
        signal = spk_model.load_audio(processed_wav_path)
        embedding = spk_model.encode_batch(signal.unsqueeze(0))
        x_vector = embedding.squeeze().cpu().numpy()
        
        print("X-Vector created")
        return x_vector.reshape(1, -1)
    except Exception as e:
        print(f"Error with extracting X-Vector: {e}")
        return None

def predict_age(x_vector, model, scaler, encoder):
    try:
        x_vector_scaled = scaler.transform(x_vector)
        prediction_encoded = model.predict(x_vector_scaled)
        prediction_label = encoder.inverse_transform(prediction_encoded)
        return prediction_label[0]
    except Exception as e:
        print(f"Error with predicting age: {e}")
        return None

def cleanup():
    if os.path.isdir(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
        print(f"Temp files '{TEMP_FOLDER}' deleted.")

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning)
    
    parser = argparse.ArgumentParser(description='Automatic age prediction from MP3 files.')
    parser.add_argument('filename', type=str, help='The name of the MP3 file located in the "input_mp3" folder.')
    args = parser.parse_args()

    speechbrain_model = load_speechbrain_model()
    if speechbrain_model is None:
        sys.exit(1)
    mp3_filename = args.filename
    input_mp3_path = os.path.join(INPUT_MP3_FOLDER, mp3_filename)
    
    if not os.path.exists(input_mp3_path):
        print(f"No mp3 file with name {input_mp3_path} found.")
        sys.exit(1)

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    base_name = os.path.splitext(mp3_filename)[0]
    temp_wav_path = os.path.join(TEMP_FOLDER, f"{base_name}_temp.wav")

    try:
        print("-" * 30)
        if not check_duration(input_mp3_path):
            sys.exit(1)
        print("-" * 30)
        if not convert_mp3_to_wav(input_mp3_path, temp_wav_path):
            sys.exit(1)
        print("-" * 30)
        if not preprocess_audio(temp_wav_path, temp_wav_path):
            sys.exit(1)
        print("-" * 30)
        x_vector = extract_x_vector(temp_wav_path, speechbrain_model)
        if x_vector is None:
            sys.exit(1)
        print("-" * 30)
        print("Loading prelearned XGBOOST model")
        try:
            model = load(MODEL_PATH)
            scaler = load(SCALER_PATH)
            encoder = load(ENCODER_PATH)
        except FileNotFoundError:
            print("One of the prelearned model files has not been found")
            sys.exit(1)
        print("Predicting")
        predicted_class = predict_age(x_vector, model, scaler, encoder)
        
        if predicted_class:
            print("\n" + "="*30)
            print(f"Age prediction for file: {mp3_filename}")
            print(f"Predicted age is:  {predicted_class}")
            print("="*30 + "\n")

    finally:
        # --- KORAK 7: Čišćenje ---
        #cleanup()
        print("\nScript has finished. ")
        input("Press Enter to exit.")