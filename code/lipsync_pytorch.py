import os

import librosa
import numpy as np
import time
import torch
import pyaudio

from scipy.signal import wiener
from collections import deque

from mfcc_extractor_lib import hard_limiter, extract_features_live, setup_logger

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
WINDOW_DURATION = 0.025
STRIDE_DURATION = 0.01

CHUNK_SIZE = int(RATE * WINDOW_DURATION)
STRIDE_SIZE = int(RATE * STRIDE_DURATION)

p = pyaudio.PyAudio()
logger = setup_logger(script_name=os.path.splitext(os.path.basename(__file__))[0])


def log_microphone_info(audio):

    try:
        default_device_info = audio.get_default_input_device_info()

        logger.info(f"Host API: {audio.get_host_api_info_by_index(default_device_info['hostApi'])['name']}")
        logger.info(f"Microphone info: {default_device_info}")

    except IOError as e:
        logger.error(f"Could not access the default input device. Error: {e}")


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

        #most_likely_prediction = torch.argmax(predictions, dim=-1).cpu().numpy()

    #return most_likely_prediction
    return predictions


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


def estimate_noise_profile(duration_sec=1):
    """Capture a short segment of audio to estimate the noise profile."""
    print("Estimating noise profile, please hold still...")
    stream = initialize_stream()
    noise_samples = []
    for _ in range(int(RATE / STRIDE_SIZE * duration_sec)):
        noise_samples.append(read_audio_chunk_to_array(stream))
    stream.stop_stream()
    stream.close()
    noise_profile = np.concatenate(noise_samples)
    print("Noise profile estimated.")
    return noise_profile


def apply_noise_reduction(buffer, noise_profile):

    if np.var(noise_profile) > 0.01:
        reduced_noise_buffer = buffer - noise_profile[:len(buffer)]
        reduced_noise_buffer = wiener(reduced_noise_buffer)
    else:
        reduced_noise_buffer = buffer - noise_profile[:len(buffer)]

    reduced_noise_buffer = np.nan_to_num(reduced_noise_buffer)
    return reduced_noise_buffer


def process_live(model, device, temporal_shift_windows=6, db_threshold=-35):
    noise_profile = estimate_noise_profile()

    predictions_buffer = deque(maxlen=temporal_shift_windows + 3)
    previous_viseme = None
    viseme_duration = 0
    buffer = np.zeros((0,), dtype=np.float32)
    prediction_counter = 0
    stream = initialize_stream()

    print("Listening... Press Ctrl+C to stop.")
    log_microphone_info(p)

    try:
        while True:
            start_time = time.time()
            audio_buffer = read_audio_chunk_to_array(stream)
            buffer = np.concatenate([buffer, audio_buffer])

            if len(buffer) >= CHUNK_SIZE:
                buffer_noisy_reduced = apply_noise_reduction(buffer[:CHUNK_SIZE], noise_profile)

                db_level = librosa.amplitude_to_db(np.abs(librosa.stft(buffer_noisy_reduced, n_fft=256)), ref=np.max).mean()
                print(f"dB Level: {db_level:.2f} dB")

                if db_level >= db_threshold:
                    limited_buffer = hard_limiter(buffer_noisy_reduced, RATE)
                    features = extract_features_live(limited_buffer, RATE)
                    predicted_viseme = make_prediction(model, device, features)
                    prediction_counter += 1

                    updated_buffer = update_predictions_buffer(predictions_buffer, predicted_viseme, prediction_counter, temporal_shift_windows)
                    if updated_buffer:
                        final_viseme, viseme_duration = process_final_viseme(predictions_buffer, previous_viseme, viseme_duration)
                        previous_viseme = final_viseme

                        elapsed_time = (time.time() - start_time) * 1000
                        yield final_viseme, elapsed_time

                    buffer = buffer[CHUNK_SIZE:]
                else:
                    elapsed_time = (time.time() - start_time) * 1000
                    print("Input below threshold, outputting 'X'.")
                    yield 'X', elapsed_time
                    buffer = buffer[CHUNK_SIZE:]

    except KeyboardInterrupt:
        print("Stream stopped")
        logger.info("Stream stopped")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def run_lipsync():

    logger.info("Attempting to find GPU...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)

    logger.info(f"Device found: {device_name}")
    logger.debug("USING CPU FOR NOW")  # Do something about this
    logger.info("Loading model...")

    model = torch.load('lipsync_model_test_three_1000ms_20bs.pth', map_location=device)

    model.to(device)

    logger.info("Model successfully loaded")
    logger.info("Starting predictions")

    for viseme, elapsed_time in process_live(model, device):
        print(f"Predicted Viseme: {viseme}, Time Elapsed: {elapsed_time:.2f} ms")  # remove this
        logger.info(f"Predicted Viseme: {viseme}, Time Elapsed: {elapsed_time:.2f} ms")

        yield viseme[0][0]

    logger.info("Process terminated")
