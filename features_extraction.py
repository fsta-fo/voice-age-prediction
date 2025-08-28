import os
import pandas as pd
import librosa
import parselmouth
import numpy as np
from tqdm import tqdm

#SETTINGS

INPUT_WAV = 'wav_files'  # Input path folder with preprocessed wav files
INPUT_TSV = 'dataset/processed_metadata.tsv' # Path to input tsv datafile
OUTPUT_TSV = 'dataset/audio_features_metadata.tsv' # Path to output extraction TSV file
USE_FEATURES_LIST = True  # IF set to True, only the list of features will be extracted, False will exract all audio parameters

#PARAMETERS

TARGET_SAMPLE_RATE = 16000
N_MFCC = 13
PITCH_FLOOR_HZ = 75
PITCH_CEILING_HZ = 500

FEATURES_LIST = [
    'f0_mean', 'f0_median', 'f0_min', 'f0_max', 'f0_std', 'f0_range',
    'jitter_local', 'shimmer_local', 'hnr',
    'formant_1', 'formant_2', 'formant_3', 'formant_4',
    'spectral_centroid_mean', 'spectral_contrast_mean', 'spectral_bandwidth_std',
    'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_8_mean', 'mfcc_10_mean', 'mfcc_13_mean'
]

#SCRIPT

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
        sound = parselmouth.Sound(audio_path)
        # F0
        pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]
        f0_features = {
            'f0_mean': np.mean(pitch_values) if len(pitch_values) > 0 else 0,
            'f0_median': np.median(pitch_values) if len(pitch_values) > 0 else 0,
            'f0_min': np.min(pitch_values) if len(pitch_values) > 0 else 0,
            'f0_max': np.max(pitch_values) if len(pitch_values) > 0 else 0,
            'f0_std': np.std(pitch_values) if len(pitch_values) > 0 else 0,
            'f0_range': (np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 0 else 0
        }
        # Jitter, Shimmer, HNR
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
        jitter_shimmer_hnr = {
            'jitter_local': parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'shimmer_local': parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'hnr': parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, PITCH_FLOOR_HZ, 0.1, 1.0)
        }
        if isinstance(jitter_shimmer_hnr['hnr'], parselmouth.Harmonicity):
            jitter_shimmer_hnr['hnr'] = parselmouth.praat.call(jitter_shimmer_hnr['hnr'], "Get mean", 0, 0)
        # Formants
        formants = sound.to_formant_burg(time_step=0.01)
        formant_features = {}
        for i in range(1, 5):
            formant_features[f'formant_{i}'] = parselmouth.praat.call(formants, "Get mean", i, 0, 0, "Hertz")
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        spectral_features = {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_contrast_std': np.std(spectral_contrast)
        }
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_features = {}
        for i in range(N_MFCC):
            mfcc_features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            mfcc_features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        all_features = {
            **f0_features, **jitter_shimmer_hnr, **formant_features, 
            **spectral_features, **mfcc_features
        }
        
        return all_features
        
    except Exception as e:
        tqdm.write(f"Error with file {audio_path}: {e}")
        return None


print("Starting extraction: ")

try:
    metadata_file = pd.read_csv(INPUT_TSV, sep='\t')
    print(f"Loaded {len(metadata_file)} rows.")
except FileNotFoundError:
    print(f"Metadata file not found: '{INPUT_TSV}'")
    exit()

all_data = []

for index, row in tqdm(metadata_file.iterrows(), total=metadata_file.shape[0], desc="Processing audio files"):
    filename = row['path']
    age = row['age']
    gender = row['gender']
    client_id = row['client_id']
    
    full_audio_path = os.path.join(INPUT_WAV, filename)

    if not os.path.exists(full_audio_path):
        tqdm.write(f"File not found: {full_audio_path}")
        continue

    features = extract_features(full_audio_path)

    if features:
        base_info = {
            'path': filename,
            'age': age,
            'gender': gender,
            'client_id' : client_id
        }
        
        if USE_FEATURES_LIST:
            selected_features = {key: features.get(key, None) for key in FEATURES_LIST}
            final_row_data = {**base_info, **selected_features}
        else:
            final_row_data = {**base_info, **features}
        all_data.append(final_row_data)


print(f"\nSaving {len(all_data)} rows")
output_df = pd.DataFrame(all_data)
output_df.to_csv(OUTPUT_TSV, index=False, encoding='utf-8')

print(f"\nFeatures saved in: {OUTPUT_TSV}")
input("Press Enter to exit.")