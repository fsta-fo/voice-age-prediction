import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import pandas as pd

#SETTINGS

INPUT_FOLDER = 'wav_files' # Input path folder with wav files
OUTPUT_FOLDER = 'processed_files' # Output path with pre-processed files
INPUT_TSV = 'dataset/sorted_metadata.tsv' # Path to input tsv datafile
OUTPUT_TSV = 'dataset/processed_metadata.tsv' # Path to updated tsv file

#PARAMETERS
TARGET_FREQUENCY = 16000
VAD_THRESHOLD_DB = 25
APPLY_FILTER = True # Set to True if you want to apply the filter
FILTER_CUTOFF_HZ = 80.0
FILTER_ORDER = 5

#SCRIPT

def audioProcessing(inputPath, outputPath):
 
    try:
        # 1. Loading
        y, sr_original = librosa.load(inputPath, sr=None)

        # 2. Resampling
        if sr_original != TARGET_FREQUENCY:
            y_resampled = librosa.resample(y, orig_sr=sr_original, target_sr=TARGET_FREQUENCY)
        else:
            y_resampled = y
        sr_nova = TARGET_FREQUENCY

        # 3. Normalization
        y_normalized = librosa.util.normalize(y_resampled)

        # 4. Voice Activity Detection (VAD)
        y_trimmed, _ = librosa.effects.trim(y_normalized, top_db=VAD_THRESHOLD_DB)

        y_final = y_trimmed
        if len(y_trimmed) == 0:
            y_final = y_normalized

        # 5. Filtering
        if APPLY_FILTER and len(y_final) > 0:
            b, a = butter(FILTER_ORDER, FILTER_CUTOFF_HZ / (0.5 * sr_nova), btype='high', analog=False)
            y_final = filtfilt(b, a, y_final)

        # 6. Saving
        sf.write(outputPath, y_final, sr_nova)

        return True

    except Exception as e:
        tqdm.write(f"Error with input file: '{os.path.basename(inputPath)}': {e}")
        return False


if __name__ == "__main__":
    print("Starting script")

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found.")
        exit()
    if not os.path.isfile(INPUT_TSV):
        print(f"Input TSV '{INPUT_TSV}' not found.")
        exit()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    inputDataTSV = pd.read_csv(INPUT_TSV, sep='\t')

    newMetadata = []
    
    for index, row in tqdm(inputDataTSV.iterrows(), total=len(inputDataTSV), desc="Working percentage: "):
        rowName = row['path']
        inputPath = os.path.join(INPUT_FOLDER, rowName)
        outputPath = os.path.join(OUTPUT_FOLDER, rowName)

        if os.path.exists(inputPath):
            if audioProcessing(inputPath, outputPath):
                newMetadata.append(row.to_dict())
        else:
            tqdm.write(f"File '{inputPath}' not found.")

    if newMetadata:
        df_new = pd.DataFrame(newMetadata)
        df_new.to_csv(OUTPUT_TSV, sep='\t', index=False, encoding='utf-8')
        print(f"\nNew metadata saved to: '{OUTPUT_TSV}'")

    print("\n Preprocessing finished. ")
    input("Press Enter to exit.")