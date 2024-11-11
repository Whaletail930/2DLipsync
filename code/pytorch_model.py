import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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


def normalize(features, stats, feature_type):
    mean, std = stats[feature_type]
    return [(x - mean) / std for x in features]


def calculate_stats(data_dir):
    mfcc_values = []
    delta_mfcc_values = []
    log_energy_values = []
    delta_log_energy_values = []

    for folder in ["TRAIN", "TEST"]:
        folder_path = os.path.join(data_dir, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                        for entry in json_data:
                            mfcc_values.extend(entry["mfcc"])
                            delta_mfcc_values.extend(entry["delta_mfcc"])
                            log_energy_values.extend(entry["log_energy"])
                            delta_log_energy_values.extend(entry["delta_log_energy"])

    stats = {
        "mfcc": (np.mean(mfcc_values), np.std(mfcc_values)),
        "delta_mfcc": (np.mean(delta_mfcc_values), np.std(delta_mfcc_values)),
        "log_energy": (np.mean(log_energy_values), np.std(log_energy_values)),
        "delta_log_energy": (np.mean(delta_log_energy_values), np.std(delta_log_energy_values))
    }

    return stats


def json_to_dataframe(json_data):
    data_list = []
    for entry in json_data:

        mfcc = normalize(entry["mfcc"], stats, "mfcc")
        delta_mfcc = normalize(entry["delta_mfcc"], stats, "delta_mfcc")
        log_energy = normalize(entry["log_energy"], stats, "log_energy")
        delta_log_energy = normalize(entry["delta_log_energy"], stats, "delta_log_energy")

        features = mfcc + delta_mfcc + log_energy + delta_log_energy
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


def data_generator(folder_path, label_encoder, frames_per_sequence, batch_size=1, steps_per_epoch=None):

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    steps = 0
    batch_features = []
    batch_labels = []

    while True:
        np.random.shuffle(file_names)

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data)

            if df.empty:
                continue

            labels_encoded = label_encoder.transform(df['label'])
            features = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').values

            num_full_sequences = len(features) // frames_per_sequence
            remainder = len(features) % frames_per_sequence

            for i in range(num_full_sequences):
                start = i * frames_per_sequence
                sequence_features = features[start:start + frames_per_sequence]
                sequence_labels = labels_encoded[start:start + frames_per_sequence]
                batch_features.append(sequence_features)
                batch_labels.append(sequence_labels)

                if len(batch_features) == batch_size:
                    steps += 1
                    yield (
                        torch.tensor(np.array(batch_features), dtype=torch.float32).to(device),
                        torch.tensor(np.array(batch_labels), dtype=torch.long).to(device)
                    )
                    batch_features = []
                    batch_labels = []

                    if steps_per_epoch and steps >= steps_per_epoch:
                        return

            if remainder > 0:
                padded_features = np.pad(features[-remainder:], ((0, frames_per_sequence - remainder), (0, 0)))
                padded_labels = np.pad(labels_encoded[-remainder:], (0, frames_per_sequence - remainder))
                batch_features.append(padded_features)
                batch_labels.append(padded_labels)

                if len(batch_features) == batch_size:
                    steps += 1
                    yield (
                        torch.tensor(np.array(batch_features), dtype=torch.float32).to(device),
                        torch.tensor(np.array(batch_labels), dtype=torch.long).to(device)
                    )
                    batch_features = []
                    batch_labels = []

                    if steps_per_epoch and steps >= steps_per_epoch:
                        return

        if batch_features:
            steps += 1
            yield (
                torch.tensor(np.array(batch_features), dtype=torch.float32).to(device),
                torch.tensor(np.array(batch_labels), dtype=torch.long).to(device)
            )
            batch_features = []
            batch_labels = []

            if steps_per_epoch and steps >= steps_per_epoch:
                return


def count_steps_per_epoch(folder_path, batch_size, sequence_length):
    total_sequences = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            df = json_to_dataframe(load_json(file_path))

            if not df.empty:
                num_sequences = len(df) // sequence_length
                remainder = len(df) % sequence_length

                total_sequences += num_sequences

                if remainder > 0:
                    total_sequences += 1

    total_steps = (total_sequences + batch_size - 1) // batch_size

    return total_steps


def plot_training_metrics(training_losses, training_accuracies, output_folder='plots'):

    os.makedirs(output_folder, exist_ok=True)

    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_folder, 'training_loss.png'))

    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_folder, 'training_accuracy.png'))

    plt.tight_layout()
    plt.show()


def train_model(model, output_folder, label_encoder, batch_size, num_epochs, learning_rate, sequence_length=1.0):
    train_folder = os.path.join(output_folder, 'TRAIN')
    frame_interval = 0.025
    frames_per_sequence = int(sequence_length / frame_interval)
    steps_per_epoch = count_steps_per_epoch(train_folder, batch_size, frames_per_sequence)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    training_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (features, labels) in enumerate(
                data_generator(train_folder, label_encoder, frames_per_sequence, batch_size, steps_per_epoch)):
            optimizer.zero_grad()
            outputs = model(features)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            if (i + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{steps_per_epoch}], Loss: {epoch_total_loss / (i + 1):.4f}')
                logger.info(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{steps_per_epoch}], Loss: {epoch_total_loss / (i + 1):.4f}')

        epoch_loss = epoch_total_loss / steps_per_epoch
        epoch_accuracy = correct_predictions / total_predictions * 100
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Training Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%')
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] Training Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%')

    print('Training complete.')

    plot_training_metrics(training_losses, training_accuracies)


def test_model(model, test_folder, label_encoder, frames_per_sequence, batch_size=200, max_examples=10000):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    examples_tested = 0

    with torch.no_grad():
        for features, labels in data_generator(test_folder, label_encoder, frames_per_sequence, batch_size, steps_per_epoch=None):
            if examples_tested >= max_examples:
                break

            outputs = model(features)
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, dim=1)
            correct = (predicted == labels).sum().item()
            total_correct += correct

            total_samples += labels.size(0)
            examples_tested += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


data_dir = r"C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT"
stats = calculate_stats(data_dir)

output_folder = r'C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT'

train_folder = os.path.join(output_folder, 'TRAIN')
label_encoder = prefit_label_encoder(train_folder)
int_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}

first_batch, _ = next(data_generator(train_folder, label_encoder, frames_per_sequence=40, batch_size=20, steps_per_epoch=1))
input_size = first_batch.shape[2]
num_visemes = len(label_encoder.classes_)
model = LipsyncModel(input_size=input_size, num_visemes=num_visemes, label_mapping=int_to_label, dropout_ratio=0.5)

train_model(model, output_folder, label_encoder, batch_size=20, num_epochs=400, learning_rate=0.001)

torch.save(model, 'lipsync_model_test_three_1000ms_20bs_e400_alt.pth')


# label_encoder = prefit_label_encoder(r"C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT\TEST")
# test_folder = os.path.join(output_folder, 'TEST')
#
# # Parameters for testing
# frames_per_sequence = int(1.0 / 0.025)  # sequence length in s is the first
# batch_size = 20 # Define batch size
#
# model = torch.load(r"C:\Users\belle\PycharmProjects\2DLipsync\code\lipsync_model_test_three_1000ms.pth")
#
# # Call the test function
# test_model(model, test_folder, label_encoder, frames_per_sequence, batch_size)
