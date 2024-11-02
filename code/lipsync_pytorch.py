import numpy as np
import time
import torch
import pyaudio

from collections import deque

from mfcc_extractor_lib import hard_limiter, extract_features_live

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
WINDOW_DURATION = 0.025
STRIDE_DURATION = 0.01

CHUNK_SIZE = int(RATE * WINDOW_DURATION)
STRIDE_SIZE = int(RATE * STRIDE_DURATION)

p = pyaudio.PyAudio()


def filter_predictions(predictions, window_size=3):
    """
    Filter noisy predictions by applying a stability check on viseme transitions.

    Parameters:
        predictions (deque): A deque holding the recent viseme predictions.
        window_size (int): Number of predictions to look ahead for stability check (default: 3).

    Returns:
        int: The filtered viseme prediction.
    """
    if len(predictions) < window_size + 1:
        return predictions[-1]

    if predictions[-1] != predictions[-2]:
        if all(pred == predictions[-1] for pred in list(predictions)[-window_size:]):
            return predictions[-1]
        else:
            return predictions[-2]
    else:
        return predictions[-1]


def remove_single_frame_visemes(current_viseme, previous_viseme, viseme_duration):
    """
    Enforce a minimum duration of two frames for each viseme.

    Parameters:
        current_viseme (int): The currently predicted viseme.
        previous_viseme (int): The previous viseme that was displayed.
        viseme_duration (int): Duration for which the current viseme has been displayed.

    Returns:
        (int, int): A tuple containing the selected viseme and updated viseme duration.
    """
    if current_viseme != previous_viseme:
        if viseme_duration < 2:
            return previous_viseme, viseme_duration + 1
        else:
            return current_viseme, 1
    else:
        return current_viseme, viseme_duration + 1


def initialize_stream():
    """Initialize and open the audio input stream."""

    return p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=STRIDE_SIZE)


def read_audio_chunk_to_array(stream):
    """Read a chunk of audio data from the stream and convert it to a numpy array"""

    audio_chunk = stream.read(STRIDE_SIZE, exception_on_overflow=False)
    audio_chunk_array = np.frombuffer(audio_chunk, dtype=np.float32)

    return audio_chunk_array


def make_prediction(model, device, features):
    """Make a prediction on the extracted features"""

    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(input_tensor.view(1, 1, -1))

    most_likely_prediction = torch.argmax(predictions, dim=-1).cpu().numpy()

    return most_likely_prediction


def update_predictions_buffer(predictions_buffer, prediction, prediction_counter, temporal_shift_windows):
    """Update the predictions buffer if enough predictions are collected."""

    if prediction_counter % 4 == 0:
        predictions_buffer.append(prediction)

    return predictions_buffer if len(predictions_buffer) > temporal_shift_windows else None


def process_final_viseme(predictions_buffer, previous_viseme, viseme_duration):
    """Filter predictions and remove single-frame visemes."""

    filtered_viseme = filter_predictions(predictions_buffer)

    if previous_viseme is None:
        previous_viseme = filtered_viseme

    final_viseme, viseme_duration = remove_single_frame_visemes(
        filtered_viseme, previous_viseme, viseme_duration
    )

    return final_viseme, viseme_duration


def process_live(model, device, temporal_shift_windows=6):

    predictions_buffer = deque(maxlen=temporal_shift_windows + 3)
    previous_viseme = None
    viseme_duration = 0
    buffer = np.zeros((0,), dtype=np.float32)
    prediction_counter = 0
    stream = initialize_stream()

    print("Listening... Press Ctrl+C to stop.")

    try:
        while True:
            start_time = time.time()

            audio_buffer = read_audio_chunk_to_array(stream)
            buffer = np.concatenate([buffer, audio_buffer])

            if len(buffer) >= CHUNK_SIZE:
                limited_buffer = hard_limiter(buffer[:CHUNK_SIZE], RATE)
                features = extract_features_live(limited_buffer, RATE)
                predicted_viseme = make_prediction(model, device, features)
                prediction_counter += 1

                updated_buffer = update_predictions_buffer(predictions_buffer, predicted_viseme, prediction_counter, temporal_shift_windows)
                if updated_buffer:
                    final_viseme, viseme_duration = process_final_viseme(predictions_buffer, previous_viseme, viseme_duration)
                    previous_viseme = final_viseme

                    elapsed_time = (time.time() - start_time) * 1000
                    yield final_viseme, elapsed_time

                buffer = buffer[STRIDE_SIZE:]

    except KeyboardInterrupt:
        print("Stream stopped")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def run_lipsync():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('lipsync_model_entire.pth', map_location=device)

    model.to(device)

    for viseme, elapsed_time in process_live(model, device):
        print(f"Predicted Viseme: {viseme}, Time Elapsed: {elapsed_time:.2f} ms")

        yield int(str(viseme).strip('[]')) #temporarily here to test animation
