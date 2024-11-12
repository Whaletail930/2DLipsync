import os
import wave

import librosa
import numpy as np
import time
import torch
import pyaudio
from scipy.signal import wiener
from collections import deque, Counter
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
    Filter noisy predictions by applying a stability check on viseme transitions with lookahead.

    Parameters:
        predictions (deque): A deque holding recent viseme predictions.
        window_size (int): Number of subsequent frames to check for stability (default: 3).

    Returns:
        int: The filtered viseme prediction, or None if filtering conditions aren’t met.
    """

    if len(predictions) < window_size + 1:
        return predictions[-1] if predictions else None

    # Lookahead-based filtering
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
    """
    if current_viseme != previous_viseme:
        if viseme_duration < 2:
            return previous_viseme, viseme_duration + 1
        else:
            return current_viseme, 1
    else:
        return current_viseme, viseme_duration + 1


def initialize_stream(audio_source='default', wav_file_path=None):
    """Initialize the audio input stream based on the selected audio source."""
    if audio_source == 'wav_file' and wav_file_path:
        wf = wave.open(wav_file_path, 'rb')
        return wf
    elif audio_source == 'default' or audio_source == 'system_audio':
        return p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=STRIDE_SIZE)
    else:
        raise ValueError("Invalid audio source specified. Choose 'default', 'system_audio', or 'wav_file'.")


def read_audio_chunk_to_array(stream, audio_source):
    """Read a chunk of audio data from the selected audio source and convert it to a numpy array."""
    if audio_source == 'wav_file':
        data = stream.readframes(STRIDE_SIZE)
        if len(data) == 0:
            return None
        audio_chunk = np.frombuffer(data, dtype=np.float32)
    else:
        audio_chunk = np.frombuffer(stream.read(STRIDE_SIZE, exception_on_overflow=False), dtype=np.float32)
    return audio_chunk


def make_prediction(model, device, features):
    """Make a prediction on the extracted features"""
    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(input_tensor.view(1, 1, -1))
    return predictions


def estimate_noise_profile(audio_source, duration_sec=1):
    """Capture a short segment of audio to estimate the noise profile."""
    print("Estimating noise profile, please hold still...")
    stream = initialize_stream()
    noise_samples = [read_audio_chunk_to_array(stream, audio_source) for _ in range(int(RATE / STRIDE_SIZE * duration_sec))]
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
    return np.nan_to_num(reduced_noise_buffer)


def downsample_predictions(predictions, original_rate=100, target_rate=24):
    """
    Downsample viseme predictions from original_rate to target_rate using majority voting.

    Parameters:
        predictions (list): List of viseme predictions at the original rate.
        original_rate (int): The original prediction rate (default is 100 Hz).
        target_rate (int): The desired output rate (default is 24 Hz).

    Returns:
        list: Downsampled viseme predictions at the target rate.
    """
    ratio = original_rate / target_rate
    downsampled_predictions = []

    for i in range(0, len(predictions), int(ratio)):
        batch = predictions[i:i + int(ratio)]
        flattened_batch = [item[0][0] if isinstance(item, list) and isinstance(item[0], list) else item
                           for item in batch]

        if not flattened_batch:
            continue

        count = Counter(flattened_batch)
        most_common_viseme, _ = count.most_common(1)[0]
        downsampled_predictions.append(most_common_viseme)

    return downsampled_predictions


def process_live(model, device, audio_source='default', wav_file_path=None, db_threshold=-35, min_silence_frames=3,
                 min_sound_frames=5):
    """Process audio data in real-time from the specified source, generating viseme predictions."""

    stream = initialize_stream(audio_source, wav_file_path)

    noise_profile = None
    if audio_source != 'wav_file':
        noise_profile = estimate_noise_profile(audio_source)

    predictions_buffer = deque(maxlen=4)
    downsample_buffer = []
    previous_viseme = 'X'
    viseme_duration = 0
    buffer = np.zeros((0,), dtype=np.float32)
    silence_mode = True
    silence_counter = 0
    sound_counter = 0

    print("Listening... Press Ctrl+C to stop.")
    log_microphone_info(p)

    try:
        while True:
            audio_capture_time = time.time()

            audio_chunk = read_audio_chunk_to_array(stream, audio_source)
            if audio_chunk is None:
                break

            buffer = np.concatenate([buffer, audio_chunk])

            if len(buffer) >= CHUNK_SIZE:
                if noise_profile is not None:
                    buffer_noisy_reduced = apply_noise_reduction(buffer[:CHUNK_SIZE], noise_profile)
                else:
                    buffer_noisy_reduced = buffer[:CHUNK_SIZE]

                db_level = librosa.amplitude_to_db(np.abs(librosa.stft(buffer_noisy_reduced, n_fft=256)),
                                                   ref=np.max).mean()
                print(f"dB Level: {db_level:.2f} dB")

                if db_level < db_threshold:
                    sound_counter = 0
                    silence_counter += 1

                    if silence_counter >= min_silence_frames:
                        silence_mode = True
                        silence_counter = min_silence_frames

                    latency = (time.time() - audio_capture_time) * 1000
                    print("Input below threshold, outputting 'X'.")
                    yield 'X', latency
                    buffer = buffer[CHUNK_SIZE:]
                    continue

                else:
                    silence_counter = 0
                    sound_counter += 1

                    if sound_counter >= min_sound_frames:
                        silence_mode = False
                        sound_counter = min_sound_frames

                if not silence_mode:
                    limited_buffer = hard_limiter(buffer_noisy_reduced, RATE)
                    features = extract_features_live(limited_buffer, RATE)
                    predicted_viseme = make_prediction(model, device, features)

                    predictions_buffer.append(predicted_viseme)
                    final_viseme = filter_predictions(predictions_buffer)
                    final_viseme = final_viseme if final_viseme is not None else previous_viseme
                    final_viseme, viseme_duration = remove_single_frame_visemes(final_viseme, previous_viseme,
                                                                                viseme_duration)
                    previous_viseme = final_viseme

                    downsample_buffer.append(final_viseme)

                    if len(downsample_buffer) >= 4:
                        downsampled_output = downsample_predictions(downsample_buffer)
                        downsample_buffer.clear()

                        for viseme in downsampled_output:
                            latency = (time.time() - audio_capture_time) * 1000
                            yield viseme, latency

                    buffer = buffer[CHUNK_SIZE:]

    except KeyboardInterrupt:
        print("Stream stopped")
        logger.info("Stream stopped")
    finally:
        if audio_source == 'wav_file':
            stream.close()
        else:
            stream.stop_stream()
            stream.close()
        p.terminate()


def run_lipsync(audio_source='default', wav_file_path=None):
    """Run the lipsync model with the selected audio source."""
    logger.info("Attempting to find GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)

    logger.info(f"Device found: {device_name}")
    logger.info("Loading model...")

    model = torch.load('lipsync_model_test_three_1000ms_20bs.pth', map_location=device)
    model.to(device)
    logger.info("Model successfully loaded")
    logger.info("Starting predictions")

    for viseme, latency in process_live(model, device, audio_source=audio_source, wav_file_path=wav_file_path):
        print(f"Predicted Viseme: {viseme}, Latency: {latency:.2f} ms")
        logger.info(f"Predicted Viseme: {viseme}, Latency: {latency:.2f} ms")

        yield viseme[0][0]

    logger.info("Process terminated")
