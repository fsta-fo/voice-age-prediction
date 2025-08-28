import pandas as pd
import random
import math
import os


#SETTINGS

INPUT_TSV = 'dataset/spectrogram_metadata.tsv' # Input path for tsv file

OUTPUT_TSV_TRAIN = 'dataset/spectrogram_metadata_train.tsv' # Output path for train tsv file
OUTPUT_TSV_TEST = 'dataset/spectrogram_metadata_test.tsv' # Output path for test tsv file

TEST_RATIO = 0.2 
RANDOM_SEED = 42

#SCRIPT

random.seed(RANDOM_SEED)

def statistics(df, dataset_name):
    print(f"Statistic for: {dataset_name}")  
    if df.empty:
        print("Dataset is empty")
        return
    print(f"Number of rows: {len(df)}")
    print("\nGender distribution:")
    print(df['gender'].value_counts())
    print("\nAge distribution:")
    print(df['age'].value_counts().sort_index())
    print("\n")


def dataset_divider():
    
    if not os.path.isfile(INPUT_TSV):
        print(f"No imput tsv found: {INPUT_TSV}")
        return
    dfWhole = pd.read_csv(INPUT_TSV, sep='\t') #for tsv sep='\t' ! for csv sep=',' 
    
    
    statistics(dfWhole, "Statistics for input tsv file)")
    print("Dividing datasets into train and test")
    
    persons = dfWhole.groupby('client_id').agg(
        age=('age', 'first'),
        gender=('gender', 'first'),
        file_count=('path', 'size')
    ).reset_index()
    
    print(f"Found {len(persons)} unique person IDs.")

    test_set_speakers_ids = list(persons[persons['file_count'] == 1]['client_id'])
    train_pool_speakers_df = persons[persons['file_count'] > 1]
    
    strata = train_pool_speakers_df.groupby(['gender', 'age'])['client_id'].apply(list).to_dict()
    train_set_speakers_ids = []
    
    for stratum_key, speakers_in_stratum in strata.items():
        random.shuffle(speakers_in_stratum)
        num_test = math.ceil(len(speakers_in_stratum) * TEST_RATIO)
        
        test_set_speakers_ids.extend(speakers_in_stratum[:num_test])
        train_set_speakers_ids.extend(speakers_in_stratum[num_test:])
        
    print(f"Division completed")
    print(f"  Training set: {len(train_set_speakers_ids)} client_id")
    print(f"  Test set: {len(test_set_speakers_ids)} client_id")

    df_train = dfWhole[dfWhole['client_id'].isin(train_set_speakers_ids)]
    df_test = dfWhole[dfWhole['client_id'].isin(test_set_speakers_ids)]
    
    df_train.to_csv(OUTPUT_TSV_TRAIN, sep='\t', index=False, encoding='utf-8')
    df_test.to_csv(OUTPUT_TSV_TEST, sep='\t', index=False, encoding='utf-8')
    
    print(f"Training set saved to: '{OUTPUT_TSV_TRAIN}'")
    print(f"Test set saved to: '{OUTPUT_TSV_TEST}'")

    print("\nStatistics:")
    statistics(df_train, "Train data")
    statistics(df_test, "Test data")


if __name__ == "__main__":
    dataset_divider()
    print("\nAll finished")
    input("Press Enter to exit.")
