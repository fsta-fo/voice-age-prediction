import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from joblib import dump
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


#SETTINGS

X_VECTORS_PATH = 'dataset/x_vectors.pkl'
TRAIN_TSV = 'dataset/audio_features_metadata_train.tsv'
TEST_TSV = 'dataset/audio_features_metadata_test.tsv'

MODEL_PATH = 'trained_models/xgboost_xvector_model.joblib'
SCALER_PATH = 'trained_models/xgboost_xvector_scaler.joblib'
ENCODER_PATH = 'trained_models/xgboost_classes.joblib'


#SCRIPT

def age_encoder(df):
    younger = ['teens', 'twenties']
    df['age'] = df['age'].replace(younger, 'twentiesAndUnder')

    older = ['sixties', 'seventies', 'eighties', 'nineties']
    df['age'] = df['age'].replace(older, '60plus')
    return df


if __name__ == "__main__":
    print("Starting the training process")

    try:
        df_xvectors = pd.read_pickle(X_VECTORS_PATH)
        df_train_info = pd.read_csv(TRAIN_TSV, sep='\t')
        df_test_info = pd.read_csv(TEST_TSV, sep='\t')
    except FileNotFoundError:
        print(f"Check path of input files!")
        exit()

    train_paths = set(df_train_info['path'])
    test_paths = set(df_test_info['path'])
    df_train = df_xvectors[df_xvectors['path'].isin(train_paths)].copy()
    df_test = df_xvectors[df_xvectors['path'].isin(test_paths)].copy()

    print(f"Data is ready")
    print(f"Training data length: {len(df_train)}")
    print(f"Testing data length: {len(df_test)}")

    df_train = age_encoder(df_train)
    df_test = age_encoder(df_test)
    print("\nUsing 5 classes")

    le = LabelEncoder()
    all_classes = pd.concat([df_train['age'], df_test['age']], ignore_index=True)
    le.fit(all_classes)
    
    df_train['age_encoded'] = le.transform(df_train['age'])
    df_test['age_encoded'] = le.transform(df_test['age'])

    y_train = df_train['age_encoded']
    y_test = df_test['age_encoded']
    
    X_train = pd.DataFrame(df_train['x_vector'].tolist())
    X_test = pd.DataFrame(df_test['x_vector'].tolist())

    print(f"\nUsing SMOTE")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Complete dataset after using SMOTE: {X_train_resampled.shape}")

#training
    print(f"\nStarting training process")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_estimators=300,
        learning_rate=0.1,
        max_depth=7,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)
    print("Training finished")
    
    y_pred = xgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining results:")
    print(f"Accuracy on the test set: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed report for trained classes:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    output_dir = os.path.dirname(MODEL_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    dump(xgb_model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    dump(le, ENCODER_PATH)

    print(f"\nModel saved in: {os.path.abspath(MODEL_PATH)}")
    print(f"Skaler saved in: {os.path.abspath(SCALER_PATH)}")
    print(f"LabelEncoder saved in: {os.path.abspath(ENCODER_PATH)}")

    print("\n Training finished. ")
    input("Press Enter to exit.")