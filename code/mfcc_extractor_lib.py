import json
import subprocess
from pathlib import Path

import numpy as np
import pyaudio
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
import os


FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK_DURATION = 0.025
CHUNK_SIZE = int(RATE * CHUNK_DURATION)
STRIDE_DURATION = 0.01
STRIDE_SIZE = int(RATE * STRIDE_DURATION)

p = pyaudio.PyAudio()

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "OUTPUT"


def hard_limiter(audio_buffer, sampling_rate, threshold=0.8):
    """
    Apply a hard limiter to the audio buffer.
    Parameters:
        audio_buffer (np.array): The audio buffer to process.
        sampling_rate (int): Sampling rate of the audio.
        threshold (float): The threshold above which audio samples will be limited.
    Returns:
        np.array: The processed audio buffer.
    """
    audio_segment = AudioSegment(
        audio_buffer.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_buffer.dtype.itemsize,
        channels=1
    )

    limited_segment = normalize(audio_segment, headroom=1.0 - threshold)

    limited_buffer = np.frombuffer(limited_segment.raw_data, dtype=audio_buffer.dtype)

    return limited_buffer


def extract_features_live(audio_buffer, sampling_rate, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40, fmax=None):

    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps

    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
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


def extract_features_from_file(file_path, sampling_rate=16000, n_mfcc=13, n_fft=256, hop_length=160, n_mels=40, fmax=None):

    audio_buffer, file_sr = librosa.load(file_path, sr=sampling_rate)

    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps

    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)

    if mfcc.shape[1] > 1:

        delta_mfcc = librosa.feature.delta(mfcc, width=3)

        log_energy = librosa.feature.rms(y=audio_buffer, frame_length=n_fft, hop_length=hop_length, center=True)
        log_energy = np.log(log_energy + np.finfo(float).eps)

        delta_log_energy = librosa.feature.delta(log_energy, width=3)

        features = [mfcc, delta_mfcc, log_energy, delta_log_energy]
    else:
        features = [mfcc, None, None, None]

    return features


def save_features_to_file(file_path, output_path_json, sampling_rate=16000, n_mfcc=13, n_fft=256,
                          hop_length=160, n_mels=40, fmax=None):

    features = extract_features_from_file(file_path, sampling_rate=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels, fmax=fmax)

    os.makedirs(os.path.dirname(output_path_json), exist_ok=True)

    time_steps = features[0].shape[1]
    features_by_time = []

    for t in range(time_steps):

        timestamp_ms = (t * hop_length / sampling_rate) * 1000

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


def process_features_file(rec_file_path):

    recording_name = Path(rec_file_path).name.strip('.wav')
    output_path_json = Path(OUTPUT_FOLDER / f"{recording_name}_features.json")

    save_features_to_file(rec_file_path, output_path_json)

    return output_path_json


def combine_data(mouthcue_file, features_file):

    file_name = Path(features_file).name.strip('_features.wav')
    # features_file = r"C:\Users\belle\PycharmProjects\2DLipsync\OUTPUT\female_1.json"
    # mouthcue_file = r"C:\Users\belle\PycharmProjects\2DLipsync\DATA\labels\female_labeled_1.txt"
    output_file = Path(OUTPUT_FOLDER / f"{file_name}_combined.json")

    pair_mouthcues_with_features(features_file, mouthcue_file, output_file)

    return output_file


def run_rhubarb_lipsync(recording_wav):

    recording_name = Path(recording_wav).name.strip('.wav')
    output_file = Path(OUTPUT_FOLDER / f"{recording_name}_visemes.json")
    rhubarb_path = r"C:\RESEARCH\2d_lipsync\Dataset generation\Rhubarb-Lip-Sync-1.13.0-Windows\rhubarb.exe"

    try:
        subprocess.run([f"{rhubarb_path}", "-f", "json", "-o", f"{output_file}", f"{recording_wav}"])

        return output_file

    except Exception as e:

        print(e)

        return None


def create_training_data(rec_file_path):

    print(f"Extracting features from {Path(rec_file_path).name}")
    features_file = process_features_file(rec_file_path)
    print("Successfully extracted features")
    print(f"Generating visemes from {Path(rec_file_path).name}")
    mouthcues_file = run_rhubarb_lipsync(Path(rec_file_path))
    print("Successfully generated visemes")
    print(f"Combining {mouthcues_file.name} and {features_file.name}")
    combined_data = combine_data(mouthcues_file, features_file)
    print("Training data successfully created")

    return combined_data


# create_training_data(r"C:\Users\belle\PycharmProjects\2DLipsync\DATA\samples\male_11_min.wav")
