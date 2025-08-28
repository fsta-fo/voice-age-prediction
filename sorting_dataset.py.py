import os
import shutil
import pandas as pd
from tqdm import tqdm
import subprocess

#SETTINGS

TO_WAV = True # If set to TRUE, wav files will be created in output folder. If FALSE only mp3 files are to be coppied
TSV_PATH = 'dataset/validated.tsv' # Path to validated.tsv file

INPUT_FOLDER = 'mp3_files' # Path to input audio folder with original mp3 files
OUTPUT_FOLDER = 'wav_files' # Path to output audio folder with cleaned mp3 or wav files
OUTPUT_TSV = 'dataset/sorted_metadata.tsv' # Path and name for newly cleaned TSV file

#Lists for cleaning up data
ageData = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties','nineties']
genderData = ['male', 'female']
rowsCleanupList = ['sentence', 'up_votes', 'down_votes', 'accent']


#SCRIPT

def ffmpegChecker():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n FFmpeg not installed or configurow")
        return False


def mp3ToWav(mp3Path, wavPath):
    try:
        cmd = [
            'ffmpeg', '-i', mp3Path, '-hide_banner', '-loglevel', 'error','-y', wavPath
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
        
    except subprocess.CalledProcessError as e:
        tqdm.write(f"  FFmpeg error with conversion {os.path.basename(mp3Path)}: {e.stderr.decode('utf-8', errors='ignore')}")
        return False
        
    except Exception as e:
        tqdm.write(f"  Unexpected error in file {os.path.basename(mp3Path)}: {e}")
        return False






if __name__ == "__main__":
    print("starting subprocess")
    if TO_WAV:
        print("Converting mpe files to wav files")
        if not ffmpegChecker():
            input("Press a key to exit")
            exit()
    else:
        print("Copying mp3 files")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        dataset_file = pd.read_csv(TSV_PATH, sep='\t')
    except FileNotFoundError:
        print(f"TSV file not found: {TSV_PATH}")
        input("Press a key to exit")
        exit()





    datasetFilterow = dataset_file[dataset_file['age'].isin(ageData) & dataset_file['gender'].isin(genderData)].copy()
    
    print(f"Found {len(datasetFilterow)} rows of data.")

    newMetaData = []

    for index, row in tqdm(datasetFilterow.iterrows(), total=len(datasetFilterow), desc="Sorting"):
        fileName = row['path']
        trueFileName = os.path.join(INPUT_FOLDER, fileName)

        if not os.path.exists(trueFileName):
            tqdm.write(f"No mp3 file with: {trueFileName} name")
            continue

        processed = False
        newFilename = ""

        if TO_WAV:
            newFilename = os.path.splitext(fileName)[0] + '.wav'
            newPath = os.path.join(OUTPUT_FOLDER, newFilename)
            if mp3ToWav(trueFileName, newPath):
                processed = True
        else:
            newFilename = fileName
            newPath = os.path.join(OUTPUT_FOLDER, newFilename)
            try:
                shutil.copy2(trueFileName, newPath)
                processed = True
            except Exception as e:
                tqdm.write(f"Error when coping file {fileName}: {e}")

        if processed:
            newRow = row.to_dict()
            newRow['path'] = newFilename
            newMetaData.append(newRow)

    if newMetaData:
        dataset_file_final = pd.DataFrame(newMetaData)
        print(f"\n Dropped rows: {rowsCleanupList}")
        dataset_file_final = dataset_file_final.drop(columns=rowsCleanupList, errors='ignore')
        dataset_file_final.to_csv(OUTPUT_TSV, sep='\t', index=False, encoding='utf-8')
        print(f"\nNew metadata for {len(dataset_file_final)} sorted for filenames in: '{OUTPUT_TSV}'")

    print("\nSuccessfully completed")
    input("Press a key to exit")