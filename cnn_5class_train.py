import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from joblib import dump


# SETTINGS

TRAIN_TSV = 'dataset\spectrogram_metadata_train.tsv'
TEST_TSV = 'dataset\spectrogram_metadata_test.tsv'
SPECTROGRAM_FOLDER = 'mel_spectrograms' #Path to mel-spectrograms folder

MODEL_FOLDER = 'cnn_model.keras'
ENCODER_FOLDER = 'cnn_label.joblib'

#PARAMETERS
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32
INITIAL_EPOH = 20
FINE_TUNE_EPOHE = 10


#SCRIPT

def path_encoder(old_spec_path):
    pic_name = os.path.basename(old_spec_path)
    return os.path.join(SPECTROGRAM_FOLDER, pic_name)

def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image, label


if __name__ == "__main__":
    
    df_train = pd.read_csv(TRAIN_TSV, sep='\t')
    df_test = pd.read_csv(TEST_TSV, sep='\t')
    
    df_train['spectrogram_path'] = df_train['spectrogram_path'].apply(path_encoder)
    df_test['spectrogram_path'] = df_test['spectrogram_path'].apply(path_encoder)

#Class encoding
    for df in [df_train, df_test]:
        older_classes = ['sixties', 'seventies', 'eighties', 'nineties']
        df['age'] = df['age'].replace(older_classes, '60plus')
        younger_classes = ['teens', 'twenties']
        df['age'] = df['age'].replace(younger_classes, 'twentiesAndUnder')
        
    print("Using 5 classes")

    le = LabelEncoder()
    all_class_e = pd.concat([df_train['age'], df_test['age']], ignore_index=True)
    le.fit(all_class_e)  
    df_train['age_encoded'] = le.transform(df_train['age'])
    df_test['age_encoded'] = le.transform(df_test['age'])
    
    dump(le, ENCODER_FOLDER)
    print(f"LabelEncoder saved in: '{ENCODER_FOLDER}'.")

    train_image_paths = df_train['spectrogram_path'].values
    train_labels = df_train['age_encoded'].values
    test_image_paths = df_test['spectrogram_path'].values
    test_labels = df_test['age_encoded'].values

    print(f"Data ready: {len(le.classes_)}")
    print("Classes:", le.classes_)


    print("\nStarting analysis")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#Start model
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
    num_classes = len(le.classes_)
    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.1), layers.RandomContrast(0.1)], name="data_augmentation")
    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)
    base_model.trainable = False
    preprocessing_layer = tf.keras.applications.efficientnet_v2.preprocess_input
    inputs = keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocessing_layer(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model structure:")
    model.summary()

#training
    print("\nStarting first phase of training:")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    history = model.fit(train_dataset, epochs=INITIAL_EPOH, validation_data=test_dataset, callbacks=[early_stopping, reduce_lr])
#Tuning
    print("\nFinetuning...")
    base_model.trainable = True
    fine_tune_at = -40
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer_fine_tune, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    total_epochs = INITIAL_EPOH + FINE_TUNE_EPOHE
    history_fine_tune = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=test_dataset, callbacks=[early_stopping, reduce_lr])
#end

    print("\nSaving and analysis")
    model.save(MODEL_FOLDER)
    print(f"Model saved in '{MODEL_FOLDER}'")
    print("\nFinal evaulation on test set:")
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Accuracy on the test set: {accuracy*100:.2f}%")
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    print("\nDetailed report for trained classes:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    
    print("\n Training finished. ")
    input("Press Enter to exit.")
