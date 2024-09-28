import json

import numpy as np
import pyaudio
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
import os
import pickle


FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK_DURATION = 0.025
CHUNK_SIZE = int(RATE * CHUNK_DURATION)
STRIDE_DURATION = 0.01
STRIDE_SIZE = int(RATE * STRIDE_DURATION)

p = pyaudio.PyAudio()


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
    audio_segment = AudioSegment(
        audio_buffer.tobytes(),
        frame_rate=sr,
        sample_width=audio_buffer.dtype.itemsize,
        channels=1
    )

    limited_segment = normalize(audio_segment, headroom=1.0 - threshold)

    limited_buffer = np.frombuffer(limited_segment.raw_data, dtype=audio_buffer.dtype)

    return limited_buffer


def extract_features_live(audio_buffer, sr, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40, fmax=None):

    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps

    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                fmax=fmax)

    if mfcc.shape[1] > 1:

        delta_mfcc = librosa.feature.delta(mfcc, width=3)

        log_energy = librosa.feature.rms(y=audio_buffer, frame_length=n_fft, hop_length=hop_length, center=True)
        log_energy = np.log(log_energy + np.finfo(float).eps)

        delta_log_energy = librosa.feature.delta(log_energy, width=3)

        features = [mfcc, delta_mfcc, log_energy, delta_log_energy]
    else:
        features = [mfcc, None, None, None]

    return features


def process_live():

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    try:
        while True:

            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_buffer = np.frombuffer(audio_chunk, dtype=np.float32)

            limited_buffer = hard_limiter(audio_buffer, RATE)

            features = extract_features_live(limited_buffer, RATE, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40,
                                             fmax=RATE / 2)

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


def extract_features_from_file(file_path, sr=16000, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40, fmax=None):

    audio_buffer, file_sr = librosa.load(file_path, sr=sr)

    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps

    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)

    if mfcc.shape[1] > 1:

        delta_mfcc = librosa.feature.delta(mfcc, width=3)

        log_energy = librosa.feature.rms(y=audio_buffer, frame_length=n_fft, hop_length=hop_length, center=True)
        log_energy = np.log(log_energy + np.finfo(float).eps)

        delta_log_energy = librosa.feature.delta(log_energy, width=3)

        features = [mfcc, delta_mfcc, log_energy, delta_log_energy]
    else:
        features = [mfcc, None, None, None]

    return features


def save_features_to_file(file_path, output_path_pickle, output_path_json, sr=16000, n_mfcc=13, n_fft=256,
                          hop_length=160, n_mels=40, fmax=None):

    features = extract_features_from_file(file_path, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels, fmax=fmax)

    os.makedirs(os.path.dirname(output_path_pickle), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_json), exist_ok=True)

    with open(output_path_pickle, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved as pickle to {output_path_pickle}")

    time_steps = features[0].shape[1]
    features_by_time = []

    for t in range(time_steps):

        timestamp_ms = (t * hop_length / sr) * 1000

        time_step_data = {
            "timestamp_ms": timestamp_ms,
            "mfcc": features[0][:, t].tolist() if features[0] is not None else None,
            "delta_mfcc": features[1][:, t].tolist() if features[1] is not None else None,
            "log_energy": features[2][:, t].tolist() if features[2] is not None else None,
            "delta_log_energy": features[3][:, t].tolist() if features[3] is not None else None
        }
        features_by_time.append(time_step_data)

    with open(output_path_json, 'w') as f:
        json.dump(features_by_time, f, indent=4)
    print(f"Features saved as JSON to {output_path_json}")


def pair_mouthcues_with_features(features_file, mouthcues_file, output_file):

    with open(features_file, 'r') as f:
        features_by_time = json.load(f)

    with open(mouthcues_file, 'r') as f:
        mouthcues_data = json.load(f)

    mouthcues = mouthcues_data["mouthCues"]

    paired_data = []

    for feature in features_by_time:
        timestamp_ms = feature["timestamp_ms"]

        mouthcue = None
        for cue in mouthcues:
            start_ms = cue["start"] * 1000
            end_ms = cue["end"] * 1000

            if start_ms <= timestamp_ms <= end_ms:
                mouthcue = cue["value"]
                break

        paired_data.append({
            "timestamp_ms": timestamp_ms,
            "mouthcue": mouthcue,
            "mfcc": feature["mfcc"],
            "delta_mfcc": feature["delta_mfcc"],
            "log_energy": feature["log_energy"],
            "delta_log_energy": feature["delta_log_energy"]
        })

    with open(output_file, 'w') as f:
        json.dump(paired_data, f, indent=4)
    print(f"Paired data saved to {output_file}")


def process_features_file():
    file_path = r"C:\RESEARCH\2d lipsync\Dataset generation\Rhubarb-Lip-Sync-1.13.0-Windows\prideandprejudice_05-06_austen_apc.wav"
    output_path = r'C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT\female_1.pkl'
    output_path_json = r'C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT\female_1.json'

    save_features_to_file(file_path, output_path, output_path_json)


def combine_data():

    features_file = r"C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT\female_1.json"
    mouthcue_file = r"C:\Users\belle\PycharmProjects\2DLipsync\DATA\labels\female_labeled_1.txt"
    output_file = r"C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT\female_1_combined.json"

    pair_mouthcues_with_features(features_file, mouthcue_file, output_file)

combine_data()