import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from LipsyncModel import LipsyncModel
from mfcc_extractor_lib import setup_logger

logger = setup_logger(script_name=os.path.splitext(os.path.basename(__file__))[0])

gpu_name = torch.cuda.get_device_name(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device('cuda'):
    logger.info(f"Using gpu: {gpu_name}")


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def json_to_dataframe(json_data):
    data_list = []
    for entry in json_data:
        features = entry["mfcc"] + entry["delta_mfcc"] + entry["log_energy"] + entry["delta_log_energy"]
        label = entry["mouthcues"][0]["mouthcue"] if len(entry["mouthcues"]) > 0 else 'X'
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


def data_generator(folder_path, batch_size=1, steps_per_epoch=None):
    label_encoder = prefit_label_encoder(folder_path)
    num_visemes = len(label_encoder.classes_)
    onehot_encoder = OneHotEncoder(sparse_output=False, categories=[np.arange(num_visemes)])

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    steps = 0

    while True:
        np.random.shuffle(file_names)
        batch_features = []
        batch_labels = []

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data)

            if df.empty:
                continue

            try:
                labels_encoded = label_encoder.transform(df['label'])

                logger.info(f"Unique labels in this file: {np.unique(labels_encoded)}")

                assert np.all(labels_encoded >= 0) and np.all(labels_encoded < num_visemes), \
                    f"Invalid label found. Labels should be in the range [0, {num_visemes - 1}]"

                features = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce')
                features = features.values.reshape((features.shape[0], 1, features.shape[1]))

                for i in range(features.shape[0]):
                    batch_features.append(features[i])
                    batch_labels.append(labels_encoded[i])

                    if len(batch_features) == batch_size:
                        steps += 1

                        batch_features_array = np.array(batch_features)
                        yield (
                            torch.tensor(batch_features_array, dtype=torch.float32).to(device),
                            torch.tensor(batch_labels, dtype=torch.long).to(device)
                        )

                        batch_features = []
                        batch_labels = []

                        if steps_per_epoch and steps >= steps_per_epoch:
                            steps = 0
                            return

            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                continue

        if batch_features:
            steps += 1

            batch_features_array = np.array(batch_features)
            yield (
                torch.tensor(batch_features_array, dtype=torch.float32).to(device),
                torch.tensor(batch_labels, dtype=torch.long).to(device)
            )
            batch_features = []
            batch_labels = []

            if steps_per_epoch and steps >= steps_per_epoch:
                steps = 0
                return


def count_steps_per_epoch(folder_path, batch_size):
    total_steps = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            df = json_to_dataframe(load_json(file_path))
            if not df.empty:
                total_steps += len(df) // batch_size
    return total_steps


def train_model(model, folder_path, batch_size, num_epochs, learning_rate, steps_per_epoch):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (features, labels) in enumerate(data_generator(folder_path, batch_size, steps_per_epoch)):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{steps_per_epoch}], Loss: {running_loss / 10:.4f}')
                logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{steps_per_epoch}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Training complete.')



folder_path = r'C:\Users\belle\PycharmProjects\2DLipsync\DATA\training'
first_batch, _ = next(data_generator(folder_path, batch_size=10, steps_per_epoch=1))
input_size = first_batch.shape[2]
num_visemes = 9
model = LipsyncModel(input_size=input_size, num_visemes=num_visemes)

steps_per_epoch = count_steps_per_epoch(folder_path, batch_size=10)
train_model(model, folder_path, batch_size=10, num_epochs=200, learning_rate=0.001, steps_per_epoch=steps_per_epoch)

torch.save(model, 'lipsync_model_test.pth')
