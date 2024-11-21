import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Function to extract features
def extract_features_live(audio_buffer, sampling_rate, n_mfcc=13, n_fft=400, hop_length=160, n_mels=128, fmax=None):
    """
    Extracts features from audio.

    Parameters:
        audio_buffer (np.ndarray): Audio buffer for feature extraction.
        sampling_rate (int): Sampling rate of the audio.
        n_mfcc (int): Number of MFCC coefficients.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for frame shifting.
        n_mels (int): Number of mel bands.
        fmax (int): Maximum frequency.

    Returns:
        dict: Features of MFCC, delta MFCC, log energy, and delta log energy.
    """
    # Normalize the audio
    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)

    # Extract delta MFCCs
    delta_width = 3
    delta_mfcc = librosa.feature.delta(mfcc, order=1, width=delta_width)

    # Extract log energy
    log_energy = librosa.feature.rms(y=audio_buffer, frame_length=n_fft, hop_length=hop_length, center=True)
    log_energy = np.log(log_energy + np.finfo(float).eps)

    # Extract delta log energy
    delta_log_energy = librosa.feature.delta(log_energy, order=1, width=delta_width)

    return {
        "mfcc": mfcc,
        "delta_mfcc": delta_mfcc,
        "log_energy": log_energy,
        "delta_log_energy": delta_log_energy
    }

# Main function to process audio
def process_audio(audio_path=None, duration=5, rate=16000, chunk=1024):
    """
    Processes audio, either from a WAV file or live recording, extracts features, and visualizes them.

    Parameters:
        audio_path (str): Path to a WAV file. If None, records audio live.
        duration (int): Duration of live recording in seconds.
        rate (int): Sampling rate in Hz.
        chunk (int): Number of frames per buffer for PyAudio.

    Returns:
        None
    """
    if audio_path:
        # Load audio from WAV file
        print(f"Loading audio from file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=rate)
        duration = librosa.get_duration(y=audio, sr=sr)
    else:
        # Record live audio
        print("Recording audio...")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

        frames = []
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Processing recorded audio...")
        audio = np.hstack(frames).astype(np.float32)
        sr = rate

    # Extract features
    features = extract_features_live(audio, sampling_rate=sr)

    # Plot the features as heatmaps
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.imshow(features["mfcc"], aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Coefficient")
    plt.title("MFCC")
    plt.ylabel("MFCC Coefficients")

    plt.subplot(4, 1, 2)
    plt.imshow(features["delta_mfcc"], aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Coefficient")
    plt.title("Delta MFCC")
    plt.ylabel("MFCC Coefficients")

    plt.subplot(4, 1, 3)
    plt.imshow(features["log_energy"], aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Value")
    plt.title("Log Energy")
    plt.ylabel("Energy")

    plt.subplot(4, 1, 4)
    plt.imshow(features["delta_log_energy"], aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Value")
    plt.title("Delta Log Energy")
    plt.ylabel("Energy")

    plt.xlabel("Frames")
    plt.tight_layout()
    plt.show()

# Example usage:
# To process a WAV file, provide the path as the `audio_path` argument
# process_audio(audio_path="path_to_your_file.wav")

# To process live audio, leave the `audio_path` argument as None

process_audio(audio_path=None, duration=5)

# process_audio(audio_path=r"C:\Users\belle\PycharmProjects\2DLipsync\DATA\TIMIT\train\dr1\mrai0\si792_284.wav")
