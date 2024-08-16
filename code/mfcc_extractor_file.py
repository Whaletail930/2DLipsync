import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import normalize


def hard_limiter(audio_buffer, sr, threshold=0.8):
    """
    Apply a hard limiter to the audio buffer.
    Parameters:
        audio_buffer (np.array): The audio buffer to process.
        sr (int): Sampling rate of the audio.
        threshold (float): The threshold above which audio samples will be limited.
    Returns:
        np.array: The processed audio buffer.
    """
    # Convert the audio buffer to pydub AudioSegment
    audio_segment = AudioSegment(
        audio_buffer.tobytes(),
        frame_rate=sr,
        sample_width=audio_buffer.dtype.itemsize,
        channels=1
    )

    # Normalize the audio to the threshold level
    limited_segment = normalize(audio_segment, headroom=1.0 - threshold)

    # Convert back to numpy array
    limited_buffer = np.frombuffer(limited_segment.raw_data, dtype=audio_buffer.dtype)

    return limited_buffer


def extract_features_live(audio_buffer, sr, n_mfcc=13, n_fft=256, hop_length=128):
    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Ensure there are enough frames to compute delta
    if mfcc.shape[1] > 1:
        # Compute the first derivative of MFCC
        delta_mfcc = librosa.feature.delta(mfcc, width=3)

        # Compute log energy
        log_energy = librosa.feature.rms(y=audio_buffer, frame_length=n_fft, hop_length=hop_length, center=True)
        log_energy = np.log(log_energy + np.finfo(float).eps)

        # Compute the first derivative of log energy
        delta_log_energy = librosa.feature.delta(log_energy, width=3)

        # Combine MFCC, delta MFCC, log energy, and delta log energy into a single feature matrix
        features = np.vstack([mfcc, delta_mfcc, log_energy, delta_log_energy])
    else:
        features = np.vstack([mfcc])

    return features


def process_audio_file(file_path, sr=16000):
    # Load the audio file
    y, _ = librosa.load(file_path, sr=sr)

    # Define chunk and stride sizes
    chunk_size = int(sr * 0.025)  # 25 ms chunks
    stride_size = int(sr * 0.01)  # 10 ms stride

    features_list = []

    # Process audio in chunks
    for start in range(0, len(y) - chunk_size + 1, stride_size):
        audio_buffer = y[start:start + chunk_size]

        # Apply hard limiter
        limited_buffer = hard_limiter(audio_buffer, sr)

        # Extract features from the audio buffer
        features = extract_features_live(limited_buffer, sr)

        # Append features to the list
        features_list.append(features)

    # Stack all features together
    all_features = np.hstack(features_list)

    return all_features


# Example usage
file_path = r'C:\Users\belle\PycharmProjects\2DLipsync\DATA\female_sample_20min.wav'
features = process_audio_file(file_path)

# Print or process the extracted features
print(features.shape)