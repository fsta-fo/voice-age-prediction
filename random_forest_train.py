import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#SETTINGS
TRAIN_PATH = 'dataset/audio_features_metadata_train.tsv'
TEST_PATH = 'dataset/audio_features_metadata_test.tsv'

MODEL_PATH = 'trained_models/radnom_forest/random_forest_age_model.joblib'
SCALER_PATH = 'trained_models/radnom_forest/scaler.joblib'
LABEL_ENCODER_PATH = 'trained_models/radnom_forest/label_encoder.joblib'


#SCRIPT
if __name__ == "__main__":

    try:
        df_train = pd.read_csv(TRAIN_PATH, sep='/t')
        df_test = pd.read_csv(TEST_PATH, sep='/t')
        print(f"Found {len(df_train)} training rows")
        print(f"Found {len(df_test)} test rows")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        exit()
    age_var = 'age' 
    le = LabelEncoder()
    df_train['age_encoded'] = le.fit_transform(df_train[age_var])
    df_test['age_encoded'] = le.transform(df_test[age_var])

    for i, class_i in enumerate(le.classes_):
        print(f"  {class_i}: {i}")

    drop_rows = ['client_id', 'path', 'age', 'gender', 'age_encoded']
    
    X_train = df_train.drop(columns=drop_rows)
    y_train = df_train['age_encoded']
    
    X_test = df_test.drop(columns=drop_rows)
    y_test = df_test['age_encoded']
    
    X_test = X_test[X_train.columns]
    print(f"\nNumber of features: {len(X_train.columns)}")

    print("\nFeatures scaling")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining model")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    print("Training finished.")
    print(f"Out-of-Bag (OOB): {rf_model.oob_score_:.4f}")

    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Label Encoder saved: {LABEL_ENCODER_PATH}")

    print("\n Training finished. ")
    input("Press Enter to exit.")