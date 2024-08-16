import librosa
import sounddevice as sd
import numpy as np

# Define parameters
sr = 44100  # Sample rate
window_size = 0.025  # Window size in seconds (25ms)
hop_size = 0.01  # Hop size in seconds (10ms)
n_mfcc = 13  # Number of MFCC coefficients to extract

# Convert window and hop sizes to samples
window_length = int(window_size * sr)
hop_length = int(hop_size * sr)

# Load the .wav file
audio_file = r"C:\Users\belle\PycharmProjects\2DLipsync\DATA\female_sample_20min.wav"
y, _ = librosa.load(audio_file, sr=sr, mono=True)

# Define callback function to process audio stream
def callback(indata, frames, time, status):
    if status:
        print(status)
    # Extract MFCCs using the loaded audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  hop_length=hop_length, n_fft=window_length)
    # Process MFCCs, e.g., save to file, perform analysis, etc.
    print(mfccs)

# Start streaming audio
with sd.InputStream(callback=callback, channels=1, samplerate=sr):
    sd.sleep(1000000)  # Adjust the sleep time as needed