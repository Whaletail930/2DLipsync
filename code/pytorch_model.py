import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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


def json_to_dataframe(json_data, stats):
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


def prefit_label_encoder(folder_path, stats):
    all_labels = set()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data, stats)
            if not df.empty:
                all_labels.update(df['label'].tolist())

    label_encoder = LabelEncoder()
    label_encoder.fit(list(all_labels))

    return label_encoder


def data_generator(folder_path, label_encoder, frames_per_sequence, batch_size=1, steps_per_epoch=None, stats=None):

    file_names = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    steps = 0
    batch_features = []
    batch_labels = []

    while True:
        np.random.shuffle(file_names)

        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data, stats)

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


def count_steps_per_epoch(folder_path, batch_size, sequence_length, stats):
    total_sequences = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            df = json_to_dataframe(load_json(file_path), stats)

            if not df.empty:
                num_sequences = len(df) // sequence_length
                remainder = len(df) % sequence_length

                total_sequences += num_sequences

                if remainder > 0:
                    total_sequences += 1

    total_steps = (total_sequences + batch_size - 1) // batch_size

    return total_steps


def validation_step(model, validation_files, label_encoder, frames_per_sequence, batch_size, stats):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        batch_features, batch_labels = [], []

        for file_path in validation_files:
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data, stats)

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

                    features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).to(device)
                    labels_tensor = torch.tensor(np.array(batch_labels), dtype=torch.long).to(device).view(-1)

                    outputs = model(features_tensor, decode=False)

                    if isinstance(outputs, list):
                        outputs = torch.tensor(outputs, device=device)

                    outputs = outputs.view(-1, outputs.size(-1))
                    loss = criterion(outputs, labels_tensor)
                    total_loss += loss.item()

                    predicted = outputs.argmax(dim=1)
                    correct_predictions += (predicted == labels_tensor).sum().item()
                    total_predictions += labels_tensor.size(0)

                    batch_features, batch_labels = [], []

            if remainder > 0:
                padded_features = np.pad(features[-remainder:], ((0, frames_per_sequence - remainder), (0, 0)))
                padded_labels = np.pad(labels_encoded[-remainder:], (0, frames_per_sequence - remainder))
                batch_features.append(padded_features)
                batch_labels.append(padded_labels)

                if len(batch_features) == batch_size:
                    features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32).to(device)
                    labels_tensor = torch.tensor(np.array(batch_labels), dtype=torch.long).to(device).view(-1)

                    outputs = model(features_tensor, decode=False)

                    if isinstance(outputs, list):
                        outputs = torch.tensor(outputs, device=device)

                    outputs = outputs.view(-1, outputs.size(-1))
                    loss = criterion(outputs, labels_tensor)
                    total_loss += loss.item()

                    predicted = outputs.argmax(dim=1)
                    correct_predictions += (predicted == labels_tensor).sum().item()
                    total_predictions += labels_tensor.size(0)

                    batch_features, batch_labels = [], []

    val_loss = total_loss / len(validation_files)
    val_accuracy = correct_predictions / total_predictions * 100
    return val_loss, val_accuracy


def run_training_process(data_dir, output_folder, model_name, sequence_length=1.0, batch_size=20, num_epochs=200, learning_rate=0.001, patience=10):
    """
    Run the training process with early stopping based on validation loss.
    """
    print('Calculating stats...')
    stats = calculate_stats(data_dir)

    train_folder = os.path.join(output_folder, 'TRAIN')
    test_folder = os.path.join(output_folder, 'TEST')
    label_encoder = prefit_label_encoder(train_folder, stats)
    int_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}

    first_batch, _ = next(data_generator(train_folder, label_encoder, frames_per_sequence=40, batch_size=20, stats=stats))
    input_size = first_batch.shape[2]
    num_visemes = len(label_encoder.classes_)
    model = LipsyncModel(input_size=input_size, num_visemes=num_visemes, label_mapping=int_to_label, dropout_ratio=0.5).to(device)

    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.json')]
    validation_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    frame_interval = 0.025
    frames_per_sequence = int(sequence_length / frame_interval)
    steps_per_epoch = count_steps_per_epoch(train_folder, batch_size, frames_per_sequence, stats)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    training_losses, validation_losses = [], []
    training_accuracies, validation_accuracies = [], []
    learning_rates = []

    print("Training init complete")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(f'{model_name}_best.pth')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        for i, (features, labels) in enumerate(
                data_generator(train_folder, label_encoder, frames_per_sequence, batch_size, steps_per_epoch, stats)):
            optimizer.zero_grad()

            outputs = model(features, decode=False)

            outputs = outputs.view(-1, outputs.size(-1)) if isinstance(outputs, torch.Tensor) else torch.tensor(outputs, device=device)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (i + 1) >= steps_per_epoch:
                break

        scheduler.step()

        train_loss = epoch_loss / steps_per_epoch
        train_accuracy = correct / total * 100
        val_loss, val_accuracy = validation_step(model, validation_files, label_encoder, frames_per_sequence, batch_size, stats)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        logger.info(f'Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saving model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    test_accuracy, confusion_mtx = test_model(model, test_files, label_encoder, frames_per_sequence, batch_size, stats)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    plot_training_metrics(training_losses, training_accuracies, validation_losses, validation_accuracies, confusion_mtx, learning_rates, label_encoder)


def test_model(model, test_files, label_encoder, frames_per_sequence, batch_size, stats):
    """
    Test the model on unseen data and compute accuracy and confusion matrix.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for file_path in test_files:
            json_data = load_json(file_path)
            df = json_to_dataframe(json_data, stats)

            if df.empty:
                continue

            labels_encoded = label_encoder.transform(df['label'])
            features = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32).to(device)
            labels_encoded = torch.tensor(labels_encoded, dtype=torch.long).to(device)

            num_full_sequences = len(features) // frames_per_sequence
            remainder = len(features) % frames_per_sequence

            for i in range(num_full_sequences):
                start = i * frames_per_sequence
                sequence_features = features[start:start + frames_per_sequence].unsqueeze(0)
                sequence_labels = labels_encoded[start:start + frames_per_sequence]

                outputs = model(sequence_features, decode=True)
                if isinstance(outputs, list):
                    decoded_outputs = [item for sublist in outputs for item in sublist]
                    predicted_indices = torch.tensor(label_encoder.transform(decoded_outputs), device=device)
                else:
                    outputs = outputs.view(-1, outputs.size(-1))
                    predicted_indices = outputs.argmax(dim=1)

                all_preds.extend(predicted_indices.tolist())
                all_labels.extend(sequence_labels.tolist())

            if remainder > 0:
                padded_features = torch.cat([
                    features[-remainder:],
                    torch.zeros(frames_per_sequence - remainder, features.size(1), device=device)
                ]).unsqueeze(0)
                padded_labels = torch.cat([
                    labels_encoded[-remainder:],
                    torch.full((frames_per_sequence - remainder,), -1, dtype=torch.long, device=device)
                ])

                outputs = model(padded_features, decode=True)
                if isinstance(outputs, list):
                    decoded_outputs = [item for sublist in outputs for item in sublist]
                    predicted_indices = torch.tensor(label_encoder.transform(decoded_outputs), device=device)
                else:
                    outputs = outputs.view(-1, outputs.size(-1))
                    predicted_indices = outputs.argmax(dim=1)

                valid_indices = padded_labels != -1
                all_preds.extend(predicted_indices[valid_indices].tolist())
                all_labels.extend(padded_labels[valid_indices].tolist())

    all_preds = torch.tensor(all_preds).cpu().numpy()
    all_labels = torch.tensor(all_labels).cpu().numpy()

    test_accuracy = (all_preds == all_labels).mean() * 100
    confusion_mtx = confusion_matrix(all_labels, all_preds, labels=range(len(label_encoder.classes_)))

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy, confusion_mtx


def plot_training_metrics(training_losses, training_accuracies, validation_losses, validation_accuracies, confusion_mtx, learning_rates, label_encoder, output_folder='plots'):
    os.makedirs(output_folder, exist_ok=True)
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'accuracy_plot.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'learning_rate_plot.png'))
    plt.close()

    decoded_labels = label_encoder.classes_
    row_normalized_confusion_mtx = confusion_mtx / confusion_mtx.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 5))
    sns.heatmap(row_normalized_confusion_mtx, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=decoded_labels,
                yticklabels=decoded_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()



data_dir = r"C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT"
output_folder = r'C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT'

run_training_process(data_dir, output_folder, 'lipsync_model_test_three_1000ms_20bs_128mel_earlystop_txt')
