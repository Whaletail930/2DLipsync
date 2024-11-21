import json
import librosa
import torch
from collections import deque, Counter
from mfcc_extractor_lib import extract_features_live, setup_logger

RATE = 16000
CHUNK_SIZE = int(RATE * 0.025)
WINDOW_SIZE = 3
logger = setup_logger("viseme_accuracy")


def load_and_expand_ground_truth(json_path):
    """
    Load and expand the ground truth viseme data so that each viseme repeats every 10ms
    for the duration it is active.
    """
    with open(json_path, 'r') as f:
        ground_truth = json.load(f)

    expanded_ground_truth = []
    for cue in ground_truth["mouthCues"]:
        start_time = cue["start"]
        end_time = cue["end"]
        value = cue["value"]

        duration_ms = int((end_time - start_time) * 1000)
        slots = duration_ms // 25

        expanded_ground_truth.extend([value] * slots)

    return expanded_ground_truth


def extract_features_from_wav(wav_path):
    """Extract features from a .wav file."""
    audio_buffer, _ = librosa.load(wav_path, sr=RATE)
    features = []

    for start in range(0, len(audio_buffer) - CHUNK_SIZE + 1, CHUNK_SIZE):
        chunk = audio_buffer[start:start + CHUNK_SIZE]
        features.append(extract_features_live(chunk, RATE))
    return features


def filter_predictions(predictions, window_size=3):
    """Filter noisy predictions by applying a stability check on viseme transitions."""
    filtered = []
    predictions_buffer = deque(maxlen=window_size)

    for pred in predictions:
        if isinstance(pred, list) and len(pred) > 0:
            pred = pred[0]

        predictions_buffer.append(str(pred))
        if len(predictions_buffer) < window_size:
            filtered.append(pred)
        else:
            counts = Counter(predictions_buffer)
            most_common = counts.most_common(1)[0][0]
            filtered.append(most_common)

    return filtered


def group_visemes_into_frames(predictions, visemes_per_frame=4):
    """
    Group viseme predictions into frames where each frame consists of a fixed number of visemes.
    Majority voting is used to determine the dominant viseme within each group,
    but all original visemes remain part of the frame.

    Parameters:
        predictions (list): List of viseme predictions.
        visemes_per_frame (int): Number of visemes per frame (default: 4 for 24Hz from 100Hz).

    Returns:
        list: List of grouped viseme predictions, where each frame retains `visemes_per_frame` visemes.
    """
    frames = []
    for i in range(0, len(predictions), visemes_per_frame):
        group = predictions[i:i + visemes_per_frame]

        if group:
            counts = Counter(group)
            most_common_viseme, _ = counts.most_common(1)[0]

            frame = [most_common_viseme] * visemes_per_frame
            frames.append(frame)

    return frames


def remove_single_frame_visemes(frames):
    """
    Remove single-frame visemes by enforcing a minimum duration of two frames per viseme.
    Each frame consists of multiple visemes, and the structure is preserved.

    Parameters:
        frames (list): List of frames, where each frame is a list of viseme predictions.

    Returns:
        list: List of frames with single-frame viseme transitions replaced.
    """
    final_frames = []
    previous_frame = None
    viseme_duration = 0

    for current_frame in frames:
        dominant_viseme = current_frame[0] if current_frame else 'X'

        if previous_frame is None or dominant_viseme != previous_frame[0]:
            if viseme_duration < 2 and previous_frame is not None:
                for _ in range(viseme_duration):
                    final_frames.append(previous_frame)
            else:
                for _ in range(max(1, viseme_duration)):
                    final_frames.append(previous_frame)

            previous_frame = current_frame or ['X'] * len(current_frame)
            viseme_duration = 1
        else:
            viseme_duration += 1

    if previous_frame is not None:
        for _ in range(max(1, viseme_duration)):
            final_frames.append(previous_frame)

    if not final_frames or final_frames[0] is None:
        final_frames = [['X']] * 8 + final_frames[1:]

    return final_frames


def downsample_and_clean_predictions(predictions, original_rate=100, target_rate=24):
    """
    Downsample predictions to match the target frame rate and clean them by replacing single-frame visemes.

    Parameters:
        predictions (list): List of raw viseme predictions at the original rate.
        original_rate (int): The original prediction rate (default: 100Hz).
        target_rate (int): The desired output rate (default: 24Hz).

    Returns:
        list: Cleaned predictions downsampled to the target frame rate.
    """

    grouped_predictions = group_visemes_into_frames(predictions, visemes_per_frame=original_rate // target_rate)

    cleaned_predictions = remove_single_frame_visemes(grouped_predictions)

    return cleaned_predictions


def make_predictions(model, device, features):
    """Make predictions using the model on the extracted features."""
    predictions = []
    for feature in features:
        input_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(input_tensor)
        predictions.append(prediction[0][0])
    return predictions


def compare_predictions(predictions, ground_truth):
    """
    Compare predictions with expanded ground truth and compute accuracy.
    Both predictions and ground truth should be lists of visemes.
    """
    total = len(ground_truth)
    correct = 0

    for i, gt_viseme in enumerate(ground_truth):
        predicted_viseme = predictions[i] if i < len(predictions) else None
        if predicted_viseme == gt_viseme:
            correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy


def flatten_predictions(predictions):
    """
    Flatten nested lists of predictions into a single list of visemes.

    Parameters:
        predictions (list): A list of predictions, which may include nested lists.

    Returns:
        list: A flattened list of viseme predictions.
    """
    flattened = []
    for frame in predictions:
        if isinstance(frame, list):
            flattened.extend(frame)
        else:
            flattened.append(frame)
    return flattened


def evaluate_wav_file(wav_file, json_file, model_path):
    """Evaluate a .wav file against its ground truth."""
    logger.info(f"Processing: {wav_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()

    features = extract_features_from_wav(wav_file)
    ground_truth = load_and_expand_ground_truth(json_file)

    raw_predictions = make_predictions(model, device, features)

    flattened_preds = [str(pred) for pred in raw_predictions]

    grouped_predictions = downsample_and_clean_predictions(flattened_preds)

    final_predictions = flatten_predictions(grouped_predictions)

    accuracy = compare_predictions(final_predictions, ground_truth)
    logger.info(f"Accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    wav_file = r"/DATA/TIMIT/train/dr1/fcjf0/sa1_0.wav"
    json_file = r"/OUTPUT/rhubarb_output/TRAIN/sa1_0_visemes.json"
    model_path = r"/code/lipsync_model_test_three_1000ms_20bs_80mel.pth"

    accuracy = evaluate_wav_file(wav_file, json_file, model_path)
    print(f"Accuracy: {accuracy:.2f}%")
