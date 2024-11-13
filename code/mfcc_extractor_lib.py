import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio
import librosa
import logging
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

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "OUTPUT"
DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"


def setup_logger(script_name):

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = DATA_DIR / "logs" / f"lipsync_{current_time}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_filename)

    formatter = logging.Formatter(
        f'%(asctime)s - {script_name} - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


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


def extract_features_live(audio_buffer, sampling_rate, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40, fmax=None):
    """
    Extracts features of 3 frames then returns the first one

    """
    audio_buffer = audio_buffer.astype(np.float32)
    audio_buffer /= np.max(np.abs(audio_buffer)) + np.finfo(float).eps

    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                                n_mels=n_mels, fmax=fmax)

    delta_width = 3

    delta_mfcc = librosa.feature.delta(mfcc, order=1, width=delta_width)

    log_energy = librosa.feature.rms(y=audio_buffer, frame_length=n_fft, hop_length=hop_length, center=True)
    log_energy = np.log(log_energy + np.finfo(float).eps)

    delta_log_energy = librosa.feature.delta(log_energy, order=1, width=delta_width)

    features = np.concatenate([mfcc, delta_mfcc, log_energy, delta_log_energy], axis=0)

    return features.T[0]


def extract_features_from_file(file_path, sampling_rate=16000, n_mfcc=13, n_fft=400, hop_length=160, n_mels=40, fmax=None):
    """
    Extract features from a .wav file after applying the hard limiter.
    """

    audio_buffer, file_sr = librosa.load(file_path, sr=sampling_rate)
    audio_buffer = hard_limiter(audio_buffer, sampling_rate)

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


def save_features_to_file(file_path, output_path_json, sampling_rate=16000, n_mfcc=13, n_fft=400,
                          hop_length=160, n_mels=40, fmax=None):

    features = extract_features_from_file(file_path, sampling_rate=sampling_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels, fmax=fmax)

    os.makedirs(os.path.dirname(output_path_json), exist_ok=True)

    audio_duration = librosa.get_duration(path=file_path, sr=sampling_rate)

    time_steps = features[0].shape[1]
    features_by_time = []

    for t in range(time_steps):
        start_timestamp_ms = (t * hop_length / sampling_rate) * 1000

        end_timestamp_ms = ((t * hop_length + n_fft) / sampling_rate) * 1000

        if end_timestamp_ms > audio_duration * 1000:
            end_timestamp_ms = audio_duration * 1000

        duration_ms = end_timestamp_ms - start_timestamp_ms

        time_step_data = {
            "start_timestamp_ms": start_timestamp_ms,
            "duration_ms": duration_ms,
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
        start_timestamp_ms = feature["start_timestamp_ms"]
        end_timestamp_ms = start_timestamp_ms + feature["duration_ms"]

        relevant_mouthcues = []
        for cue in mouthcues:
            start_ms = cue["start"] * 1000
            end_ms = cue["end"] * 1000

            if start_ms < end_timestamp_ms and end_ms > start_timestamp_ms:
                relevant_mouthcues.append({
                    "mouthcue": cue["value"],
                    "start": start_ms,
                    "end": end_ms
                })

        paired_data.append({
            "start_timestamp_ms": start_timestamp_ms,
            "duration_ms": feature["duration_ms"],
            "mouthcues": relevant_mouthcues,
            "mfcc": feature["mfcc"],
            "delta_mfcc": feature["delta_mfcc"],
            "log_energy": feature["log_energy"],
            "delta_log_energy": feature["delta_log_energy"]
        })

    with open(output_file, 'w') as f:
        json.dump(paired_data, f, indent=4)
    print(f"Paired data saved to {output_file}")


def process_features_file(rec_file_path, data_type):

    recording_name = Path(rec_file_path).name.strip('.wav')
    output_path_json = Path(OUTPUT_DIR / "features" / data_type / f"{recording_name}_features.json")

    save_features_to_file(rec_file_path, output_path_json)

    return output_path_json


def combine_data(mouthcue_file, features_file, data_type):

    file_name = Path(features_file).name.strip('_features.wav')

    output_file = Path(OUTPUT_DIR / data_type / f"{file_name.strip('.json')}_combined.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pair_mouthcues_with_features(features_file, mouthcue_file, output_file)

    return output_file


def run_rhubarb_lipsync(recording_wav, data_type):

    recording_name = Path(recording_wav).name.strip('.wav')
    output_file = Path(OUTPUT_DIR / "rhubarb_output" / data_type / f"{recording_name}_visemes.json")
    rhubarb_path = r"C:\RESEARCH\2d_lipsync\Dataset generation\Rhubarb-Lip-Sync-1.13.0-Windows\rhubarb.exe"

    try:
        subprocess.run([f"{rhubarb_path}", "-f", "json", "-o", f"{output_file}", f"{recording_wav}"])

        return output_file

    except Exception as e:

        print(e)

        return None


def create_training_data(rec_file_path, data_type):

    print(f"Extracting features from {Path(rec_file_path).name}")
    features_file = process_features_file(rec_file_path, data_type)
    print("Successfully extracted features")
    print(f"Generating visemes from {Path(rec_file_path).name}")
    mouthcues_file = run_rhubarb_lipsync(Path(rec_file_path), data_type)
    print("Successfully generated visemes")
    print(f"Combining {mouthcues_file.name} and {features_file.name}")
    combined_data = combine_data(mouthcues_file, features_file, data_type)
    print("Training data successfully created")

    return combined_data


def process_wav_files(base_dir, folder_type):
    """
    Process only .wav files within the specified folder, converting .WAV files in place using sph2pipe.
    """

    target_dir = Path(base_dir) / folder_type
    processed_count = 0

    for root, subdirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.wav'):
                new_file = f"{str(file).strip('.wav')}_{processed_count}.wav"
                new_file_path = Path(root) / new_file
                file_path = Path(root) / file
                os.rename(file_path, new_file_path)
                logger.info(f"Processing {file_path}")

                create_training_data(new_file_path, folder_type)

                print(f"Processed and converted {new_file_path}")
                processed_count += 1

    print(f"All done. Files processed: {processed_count}")


# #model = tf.keras.models.load_model(r'C:\Users\belle\PycharmProjects\2DLipsync\code\lipsync_model')
# model = keras.models.load_model(r'C:\Users\belle\PycharmProjects\2DLipsync\code\lipsync_model_2')
# print(model.input_shape)
# print(model.summary())
# #model.load_weights(r"C:\Users\belle\PycharmProjects\2DLipsync\code\lipsync_model.h5")
# #model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
# process_live(model)


# logger = setup_logger(script_name=os.path.splitext(os.path.basename(__file__))[0])
# process_wav_files(r"C:\Users\belle\PycharmProjects\2DLipsync\DATA\TIMIT", "train")
