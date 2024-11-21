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
ORIGINAL_RATE = 100
TARGET_RATE = 24
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

    if predictions[-1] != predictions[-2]:
        if all(pred == predictions[-1] for pred in list(predictions)[-window_size:]):
            return predictions[-1]
        else:
            return predictions[-2]
    else:
        return predictions[-1]


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
        # Group the next `visemes_per_frame` visemes
        group = predictions[i:i + visemes_per_frame]

        flattened_group = [item for sublist in group for innerlist in sublist for item in innerlist]
        # Apply majority voting to determine the dominant viseme in the group
        counts = Counter(flattened_group)
        most_common_viseme, _ = counts.most_common(1)[0]

        # Replace all visemes in the group with the most common one
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
        # Ensure the frame structure is consistent (all elements in a frame are the same)
        dominant_viseme = current_frame[0] if current_frame else 'X'

        if previous_frame is None or dominant_viseme != previous_frame[0]:
            if viseme_duration < 2 and previous_frame is not None:
                # If the previous viseme was too short, replace it with the current viseme
                final_frames.extend([previous_frame] * viseme_duration)
            viseme_duration = 2
            previous_frame = current_frame
        else:
            # Increment the duration for the current viseme
            viseme_duration += 1

        # Add the current frame to the final list if it has stabilized
        if viseme_duration >= 2:
            final_frames.append(current_frame)

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
    # Group predictions into frames
    grouped_predictions = group_visemes_into_frames(predictions, visemes_per_frame=original_rate // target_rate)

    # Replace single-frame visemes from the grouped predictions
    cleaned_predictions = remove_single_frame_visemes(grouped_predictions)

    return cleaned_predictions


def process_live(model, device, db_threshold=-35, audio_source='default', wav_file_path=None, min_silence_frames=5,
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

            # Read audio chunk
            audio_chunk = read_audio_chunk_to_array(stream, audio_source)
            if audio_chunk is None:
                break

            buffer = np.concatenate([buffer, audio_chunk])

            if len(buffer) >= CHUNK_SIZE:
                if noise_profile is not None:
                    buffer_noisy_reduced = apply_noise_reduction(buffer[:CHUNK_SIZE], noise_profile)
                else:
                    buffer_noisy_reduced = buffer[:CHUNK_SIZE]

                # Process audio
                db_level = librosa.amplitude_to_db(np.abs(librosa.stft(buffer_noisy_reduced, n_fft=256)),
                                                   ref=np.max).mean()

                if db_level < db_threshold:
                    silence_counter += 1
                    if silence_counter >= min_silence_frames:
                        silence_mode = True
                        silence_counter = min_silence_frames

                    latency = (time.time() - audio_capture_time) * 1000
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
                    previous_viseme = final_viseme

                    downsample_buffer.append(final_viseme)

                    # Process buffer for grouped predictions
                    if len(downsample_buffer) >= ORIGINAL_RATE // TARGET_RATE:
                        grouped_predictions = group_visemes_into_frames(
                            downsample_buffer,
                            visemes_per_frame=ORIGINAL_RATE // TARGET_RATE
                        )
                        downsample_buffer.clear()

                        cleaned_predictions = remove_single_frame_visemes(grouped_predictions)

                        for frame in cleaned_predictions:
                            for viseme in frame:
                                latency = (time.time() - audio_capture_time) * 1000
                                yield viseme, latency

                    buffer = buffer[CHUNK_SIZE:]

    except KeyboardInterrupt:
        print("Stream stopped")
        logger.info("Stream stopped")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def run_lipsync(db_threshold, audio_source='default', wav_file_path=None):
    """Run the lipsync model with the selected audio source."""
    logger.info("Attempting to find GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)

    logger.info(f"Device found: {device_name}")
    logger.info("Loading model...")

    model = torch.load('lipsync_model_test_three_1000ms_20bs_re2.pth', map_location=device)
    model.to(device)
    logger.info("Model successfully loaded")
    logger.info("Starting predictions")

    for viseme, latency in process_live(model, device, db_threshold, audio_source=audio_source, wav_file_path=wav_file_path):
        print(f"Predicted Viseme: {viseme}, Latency: {latency:.2f} ms")
        logger.info(f"Predicted Viseme: {viseme}, Latency: {latency:.2f} ms")

        yield viseme[0][0]

    logger.info("Process terminated")