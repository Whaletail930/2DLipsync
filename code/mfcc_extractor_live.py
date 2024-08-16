import numpy as np
import pyaudio
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


def extract_features_live(audio_buffer, sr, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40, fmax=None):
    # Ensure the audio buffer is in float32 format and normalize it
    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps  # Normalize to avoid overflow

    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                fmax=fmax)

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
        features = [mfcc, delta_mfcc, log_energy, delta_log_energy]
    else:
        features = [mfcc, None, None, None]

    return features


# Audio stream configuration
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000  # Sample rate in Hz
CHUNK_DURATION = 0.025  # Each chunk duration in seconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION)  # Each chunk size in samples
STRIDE_DURATION = 0.01  # Stride duration in seconds
STRIDE_SIZE = int(RATE * STRIDE_DURATION)  # Stride size in samples

p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,  # This line specifies that the input is from the microphone
                frames_per_buffer=CHUNK_SIZE)

try:
    while True:
        # Read audio stream
        audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_buffer = np.frombuffer(audio_chunk, dtype=np.float32)

        # Apply hard limiter
        limited_buffer = hard_limiter(audio_buffer, RATE)

        # Extract features from the audio buffer
        features = extract_features_live(limited_buffer, RATE, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40,
                                         fmax=RATE / 2)

        # Print the actual features
        mfcc, delta_mfcc, log_energy, delta_log_energy = features
        print("MFCC:\n", mfcc)
        if delta_mfcc is not None:
            print("Delta MFCC:\n", delta_mfcc)
        if log_energy is not None:
            print("Log Energy:\n", log_energy)
        if delta_log_energy is not None:
            print("Delta Log Energy:\n", delta_log_energy)

except KeyboardInterrupt:
    print("Stream stopped")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
