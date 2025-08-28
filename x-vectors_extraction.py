
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import warnings

try:
    from speechbrain.pretrained import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
except ImportError:
    print("Please install speechbrain library")
    exit()

#SETTINGS

INPUT_TSV = 'dataset/processed_metadata.tsv' # Path to input processed tsv datafile
INPUT_FOLDER = 'processed_files' # Input path folder with processed wav files
PICKLE_OUTPUT = 'dataset/x_vectors.pkl' # Path to output pickle (vectors) file 
SAVEDIR_MODEL = 'pretrained_vector/spkrec-xvect-voxceleb' # Path used for VoxCeleb model

#SCRIPT

def open_model():
    try:
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir=SAVEDIR_MODEL,
            local_strategy=LocalStrategy.COPY
        )
        print("Model fully loaded")
        return spk_model
    except Exception as e:
        print(f"Error reading model: {e}")
        return None

def x_vector_generator(data_file_m, model):
    result = []
    print("\nStarting extraction")
    for index, row in tqdm(data_file_m.iterrows(), total=len(data_file_m), desc="Extracting percentage:"):
        wav_path = os.path.join(INPUT_FOLDER, row['path'])
        if not os.path.exists(wav_path):
            tqdm.write(f"File not found with name: {wav_path}")
            continue          
        try:
            signal = model.load_audio(wav_path)
            embedding = model.encode_batch(signal.unsqueeze(0))
            embedding = embedding.squeeze().cpu().numpy()
            row_dict = row.to_dict()
            row_dict['x_vector'] = embedding
            result.append(row_dict)           
        except Exception as e:
            tqdm.write(f"Error when extracting vectors for: {row['path']}: {e}")           
    return result


if __name__ == "__main__":
    
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    try:
        file_data_no = pd.read_csv(INPUT_TSV, sep='\t')
        print(f"Found {len(file_data_no)} rows in '{os.path.basename(INPUT_TSV)}'")
    except FileNotFoundError:
        print(f"No input TSV file found:: {INPUT_TSV}")
        input("Press Enter to exit.")
        exit()
 
    model = open_model()
    
    if model:
        all_results = x_vector_generator(file_data_no, model)
        if all_results:
            df_xvectors = pd.DataFrame(all_results)
            df_xvectors.to_pickle(PICKLE_OUTPUT)
            print(f"\nExtraction finished for {len(all_results)} files.")
            print(f"X-Vectors created in: '{PICKLE_OUTPUT}'")
        else:
            print("\nMajor error")

    print("\nAll finished")
    input("Press Enter to exit.")
