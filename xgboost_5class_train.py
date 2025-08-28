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

TRAIN_DATASET = 'dataset/audio_features_metadata_train.tsv'
TEST_DATASET = 'dataset/audio_features_metadata_test.tsv'

OUTPUT_MODEL_PATH = 'trained_models/xgboost_model.joblib'
OUTPUT_SCALER_PATH = 'trained_models/xgboost_scaler.joblib'
OUTPUT_ENCODER_PATH = 'trained_models/xgboost_age_encoder.joblib'

#SCRIPT
if __name__ == "__main__":

    try:
        df_train = pd.read_csv(TRAIN_DATASET, sep='\t')
        df_test = pd.read_csv(TEST_DATASET, sep='\t')
    except FileNotFoundError:
        print(f"Input file not found")
        exit()
        
    print("Age class encoding")
    
# AGE CLASS ENCODING
    younger = ['teens', 'twenties']
    df_train['age'] = df_train['age'].replace(younger, 'twentiesAndUnder')
    df_test['age'] = df_test['age'].replace(younger, 'twentiesAndUnder')

    older = ['sixties', 'seventies', 'eighties', 'nineties']
    df_train['age'] = df_train['age'].replace(older, '60plus')
    df_test['age'] = df_test['age'].replace(older, '60plus')
#END

    le = LabelEncoder()
    df_train['age_encoded'] = le.fit_transform(df_train['age'])
    df_test['age_encoded'] = le.transform(df_test['age'])

    y_train = df_train['age_encoded']
    y_test = df_test['age_encoded']

    drop_columns = ['path', 'age', 'gender', 'client_id', 'age_encoded']
    existing_drop_columns = [col for col in drop_columns if col in df_train.columns]
    
    X_train = df_train.drop(columns=existing_drop_columns)
    X_test = df_test.drop(columns=existing_drop_columns)
    X_test = X_test[X_train.columns]
    
    print(f"All data ready: {X_train.shape[1]}")

    print(f"Using SCALER and SMOTE")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"SMOTE training size: {X_train_resampled.shape}")


    print(f"\nTraining the model")
    
    
#MODEL PARAMETERS
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
    print("Model has been trained")

    print(f"\nModel score:")
    y_pred = xgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClass report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    

    
    print(f"\nSaving final model: ")
    output_dir = os.path.dirname(OUTPUT_MODEL_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dump(xgb_model, OUTPUT_MODEL_PATH)
    dump(scaler, OUTPUT_SCALER_PATH)
    dump(le, OUTPUT_ENCODER_PATH)
    
    print(f"\nModel has been successfully generated in: {os.path.abspath(OUTPUT_MODEL_PATH)}")
    print(f"Skaler has been successfully generated in: {os.path.abspath(OUTPUT_SCALER_PATH)}")
    print(f"LabelEncoder has been successfully generated in: {os.path.abspath(OUTPUT_ENCODER_PATH)}")

    print("\n Training finished. ")
    input("Press Enter to exit.")