import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import multiprocessing

#SETTINGS
INPUT_FOLDER = 'processed_files' # Input path folder with processed wav files
INPUT_TSV = 'dataset/processed_metadata.tsv' # Path to input processed tsv datafile
OUTPUT_TSV = 'dataset/spectrogram_metadata.tsv' # Path to updated tsv file
FOLDER_SPECTROGRAMS = 'mel_spectrograms' # Path location for Mel-spectrogram images
SR = 16000 # Sample Rate

#SCRIPT


def spectrogram_generator(y, sr, out_path):
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        tqdm.write(f"Error generating spectrogram image for file: {os.path.basename(out_path)}: {e}")
        return False



def file_process(row):
    index, data, columns = row
    row_dict = dict(zip(columns, data))
    wav_filename = row_dict['path']
    input_path = os.path.join(INPUT_FOLDER, wav_filename)

    if os.path.exists(input_path):
        try:
            y, sr_native = librosa.load(input_path, sr=SR)
            if len(y) > 0:
                img_name = os.path.splitext(wav_filename)[0]
                spectrogram_path = os.path.join(FOLDER_SPECTROGRAMS, f"{img_name}.png")
                success = spectrogram_generator(y, SR, spectrogram_path)
                
                if success:
                    f_row = row_dict
                    f_row['spectrogram_path'] = spectrogram_path
                    return f_row
                else:
                    return None
            else:
                return f"Wav file is empty: {wav_filename}"
        except Exception as e:
            return f"Error with file: '{wav_filename}': {e}"
    else:
        return f"File not found: {input_path}"

if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning) 
    
    print("Generating mel-spectrograms")
    os.makedirs(FOLDER_SPECTROGRAMS, exist_ok=True)
    
    if not os.path.isdir(INPUT_FOLDER):
        print(f"No input wav folder found: {INPUT_FOLDER}")
        exit()
    if not os.path.isfile(INPUT_TSV):
        print(f"No input metadata file found: {INPUT_TSV}")
        exit()

    metadata = pd.read_csv(INPUT_TSV, sep='\t')
    tasks = [(index, row, metadata.columns) for index, row in metadata.iterrows()]
    p_rows = []
    num_processes = multiprocessing.cpu_count()
    print(f"Using multicore: {num_processes} CPU cores will be used")

    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap_unordered(file_process, tasks), total=len(tasks), desc="Generating Mel-spectrograms"):
                if isinstance(result, dict):
                    p_rows.append(result)
                elif isinstance(result, str):
                    tqdm.write(result)
    
    except Exception as e:
        print(f"Error: {e}")

    finally:
        if p_rows:
            print(f"\nSaving {len(p_rows)} rows")
            df_processed = pd.DataFrame(p_rows)
            original_colums = list(metadata.columns)
            final_columns = original_colums + ['spectrogram_path']
            df_processed = df_processed[[col for col in final_columns if col in df_processed.columns]]
            df_processed.to_csv(OUTPUT_TSV, sep='\t', index=False, encoding='utf-8')
            print(f"Finished, all data saved in:  '{OUTPUT_TSV}'")
        else:
            print("\nNo data!")

    print("\nAll finished")
    input("Press Enter to exit.")
