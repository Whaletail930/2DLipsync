import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
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


def data_generator(folder_path, batch_size=32):
    label_encoder = None
    onehot_encoder = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data)

            labels, label_encoder, onehot_encoder = encode_labels(df, label_encoder, onehot_encoder)

            features = df.drop(columns=['label', 'label_encoded']).values
            timesteps = 1
            features = features.reshape((features.shape[0], timesteps, features.shape[1]))

            for start in range(0, features.shape[0], batch_size):
                end = min(start + batch_size, features.shape[0])
                yield features[start:end], labels[start:end]


def create_model(input_shape, num_visemes):
    lipsync_model = Sequential()
    lipsync_model.add(LSTM(200, input_shape=input_shape))
    lipsync_model.add(Dropout(0.5))
    lipsync_model.add(Dense(num_visemes, activation='softmax'))
    lipsync_model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return lipsync_model


folder_path = r'C:\Users\belle\PycharmProjects\2DLipsync\DATA\training'

first_batch = next(data_generator(folder_path))
input_shape = (first_batch[0].shape[1], first_batch[0].shape[2])
num_visemes = 9  # Update code to make this dynamic (possibly issue at label encoding)

model = create_model(input_shape, num_visemes)
model.build(input_shape=(None, input_shape[0], input_shape[1]))
model.summary()

steps_per_epoch = sum([len(pd.read_json(os.path.join(folder_path, f))) for f in os.listdir(folder_path) if f.endswith('.json')]) // 32

model.fit(
    data_generator(folder_path),
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    batch_size=10,
    validation_data=data_generator(folder_path),
    validation_steps=steps_per_epoch // 10
)

X_val, y_val = next(data_generator(folder_path))
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
