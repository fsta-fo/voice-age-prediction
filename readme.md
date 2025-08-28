Age estimation from voice

This project was developed as a part of the thesis titled "Procjena dobi osoba na temelju glasa" (Age estimation of persons based on voice).
Supervisor: prof. dr. sc. Petra Grd

Project overview
Repository contains a collection of python scripts designed to process audio data, extract features, train models, and predict the age group of a speaker from a voice sample.

Code serves as a testing ground, utilizing a small subset of 500 audio files from the publicly available Common Voice Corpus 4 (English).

Pre-trained Models
The final, pre-trained models, which were trained on the entire dataset of 503,339 audio files, can be found in directory trained_models/full_trained/

Prerequisites:

1. Python 3.10

2. FFmpeg
	- Mandatory for converting audio files from MP3 to WAV format.

3. Python Libraries
	- Listed in the requirements.txt file. To install them, use pip install -r requirements.txt (best used when done in virtual enviroment)
	

The project is organized into modular Python scripts and typical workflow for processing data and training models is outlined below.
When using CSV or TSV files, you may need to adjust the delimiter in the pd.read_csv() function. For example, change sep='\t' #(for TSV) to sep=',' #(for CSV).

1. sorting_dataset.py
	- Cleans the initial dataset metadata (validated.tsv) to include only files with known age and gender. It converts the source MP3 files to WAV format.
	- Creates the wav_files directory and a cleaned dataset/sorted_metadata.tsv file.

2. preprocessing_wav.py
	- Applies a series of preprocessing steps to the WAV files, including normalization, voice activity detection (VAD), and filtering.
	- Creates new datase: dataset/processed_metadata.tsv and new directory processed_files containing the processed WAVs.

3. features_extraction.py
	- Extracts traditional audio features (like F0, jitter, shimmer, MFCCs) from the processed WAV files
	- Creates new dataset file: dataset/audio_features_metadata.tsv

4. spectrogram_extraction.py
	- Generates Mel-spectrogram images from the processed WAV files.
	- Creates new dataset: dataset/spectrogram_metadata.tsv and a new directory mel_spectrograms containing spectrogram images.

5. x-vectors_extraction.py
	- Extracts x-vectors using a pre-trained model ( SpeechBrain v1.0.3 using the speechbrain/spkrec-xvect-voxceleb model).
	- Creates vectors dataset: dataset/x_vectors.pkl
	- Local cache of the model is in pretrained_vector/spkrec-xvect-voxceleb.

6. train_test_divider.py
	- Splits a metadata file into training and testing sets based on client_id to ensure speaker independence. It can process either audio_features_metadata.tsv or spectrogram_metadata.tsv.
	- Creates sliced metadata files: audio_features_metadata_train.tsv and audio_features_metadata_test.tsv or spectrogram_metadata_train.tsv and spectrogram_metadata_test.tsv in dataset folder

7. random_forest_train.py
	- Trains a Random forest model on all age classes using the traditional audio features.
	- Creates trained_models/random_forest_age_model.joblib, trained_models/scaler.joblib and trained_models/label_encoder.joblib model files

8. xgboost_5class_train.py
	- Trains an XGBoost model on 5 merged age classes using traditional audio features.
	- Creates trained_models/xgboost_model.joblib, trained_models/xgboost_scaler.joblib and trained_models/ xgboost_age_encoder.joblib model files

9. cnn_5class_train.py
	- Trains a Convolutional Neural Network (CNN) on 5 merged age classes using the generated Mel spectrograms.
	- Creates trained_models/cnn_model.keras and trained_models/cnn_label.joblib model files

10. xgboost_5class_train_vectors.py
	- Trains an XGBoost model on 5 merged age classes using the extracted x-vectors.
	- Creates trained_models/xgboost_xvector_model.joblib, trained_models/xgboost_xvectors_saler.joblib and trained_models/xgboost_classes.joblib model files

11. standalone_prediction.py
	- A standalone script that uses the best-performing trained model to predict the age group for a single input MP3 file (max 10 seconds long). It performs the entire preprocessing and feature extraction pipeline internally.
	- Outputs predicted age class (twentiesAndUnder, thirties, fourties, fifties or 60plus)