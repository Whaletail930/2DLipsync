import pyaudio
import wave
import numpy as np
import librosa

FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono audio
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 1024              # 2^10 samples per frame
OUTPUT_FILENAME = "../output.wav"


def record_audio(duration=10, output_filename=OUTPUT_FILENAME):
    """Records audio for the specified duration and saves it to a file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {output_filename}")


def monitor_db_level():
    """Continuously monitors and prints the dB level of live audio input."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Monitoring dB levels... Press Ctrl+C to stop.")

    try:
        buffer = np.zeros((0,), dtype=np.float32)  # Initialize an empty buffer with float32 type
        while True:
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)  # Convert to float32
            buffer = np.concatenate([buffer, audio_data])  # Add new audio data to the buffer

            # Only calculate dB level if buffer has enough data
            if len(buffer) >= CHUNK:
                stft = librosa.stft(buffer[:CHUNK])
                db_level = librosa.amplitude_to_db(np.abs(stft), ref=np.max).mean()
                print(f"dB Level: {db_level:.2f} dB")

                # Remove processed chunk from the buffer
                buffer = buffer[CHUNK:]

    except KeyboardInterrupt:
        print("Stopped monitoring.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def get_wavefile_db_levels(file_path):
    """Outputs the dB levels of a wave file for each time frame."""

    y, sr = librosa.load(file_path, sr=None)
    stft = librosa.stft(y, n_fft=256)
    db_levels = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    for i, frame_db in enumerate(db_levels.T):
        avg_db = frame_db.mean()
        print(f"Frame {i+1}: {avg_db:.2f} dB")

    return db_levels


get_wavefile_db_levels("../output.wav")
