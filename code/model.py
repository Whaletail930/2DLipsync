import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Masking
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.layers import LSTM, Dense, Dropout


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def json_to_dataframe(json_data):
    data_list = []
    for entry in json_data:
        features = entry["mfcc"] + entry["delta_mfcc"] + entry["log_energy"] + entry["delta_log_energy"]
        if len(entry["mouthcues"]) > 0:
            label = entry["mouthcues"][0]["mouthcue"]
        else:
            label = 'X'
        data_list.append(features + [label])

    num_features = (
        len(json_data[0]["mfcc"]) +
        len(json_data[0]["delta_mfcc"]) +
        len(json_data[0]["log_energy"]) +
        len(json_data[0]["delta_log_energy"])
    )
    columns = [f'feature_{i}' for i in range(num_features)] + ['label']
    df = pd.DataFrame(data_list, columns=columns)
    return df


def encode_labels(df, label_encoder=None, onehot_encoder=None):
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['label'])
    else:
        df['label_encoded'] = label_encoder.transform(df['label'])

    if onehot_encoder is None:
        onehot_encoder = OneHotEncoder(sparse_output=False)
        labels_onehot = onehot_encoder.fit_transform(df[['label_encoded']])
    else:
        labels_onehot = onehot_encoder.transform(df[['label_encoded']])

    return labels_onehot, label_encoder, onehot_encoder


def prefit_label_encoder(folder_path):
    all_labels = set()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data)

            if not df.empty:
                all_labels.update(df['label'].tolist())

    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_labels))

    return label_encoder


def data_generator(folder_path, batch_size=10):
    label_encoder = prefit_label_encoder(folder_path)
    num_visemes = len(label_encoder.classes_)
    onehot_encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(num_visemes)])

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    total_files = len(file_names)

    while True:
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data)

            if df.empty:
                continue

            try:
                labels_encoded = label_encoder.transform(df['label'])
                labels_onehot = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))

                features = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce')
                features = features.values.reshape((features.shape[0], 1, features.shape[1]))  # Reshape for LSTM

                for start in range(0, features.shape[0], batch_size):
                    end = min(start + batch_size, features.shape[0])
                    yield features[start:end], labels_onehot[start:end]

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue


def count_steps_per_epoch(folder_path, batch_size):
    total_steps = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            df = json_to_dataframe(load_json(file_path))
            if not df.empty:
                total_steps += len(df) // batch_size
    return total_steps


def create_model(input_shape, num_visemes):
    lipsync_model = Sequential()
    lipsync_model.add(Masking(mask_value=0.0, input_shape=input_shape))
    lipsync_model.add(LSTM(200))
    lipsync_model.add(Dropout(0.5))
    lipsync_model.add(Dense(num_visemes, activation='softmax'))
    lipsync_model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return lipsync_model


folder_path = r'C:\Users\belle\PycharmProjects\2DLipsync\DATA\training'

first_batch = next(data_generator(folder_path))
input_shape = (first_batch[0].shape[1], first_batch[0].shape[2])
num_visemes = first_batch[1].shape[1]


model = create_model(input_shape, num_visemes)
model.build(input_shape=(None, input_shape[0], input_shape[1]))
model.summary()

steps_per_epoch = count_steps_per_epoch(folder_path, batch_size=10)

model.fit(
    data_generator(folder_path),
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    validation_data=data_generator(folder_path),
    validation_steps=steps_per_epoch // 10
)


X_val, y_val = next(data_generator(folder_path))
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

model.save('lipsync_model')
model.save('lipsync_model.h5')
