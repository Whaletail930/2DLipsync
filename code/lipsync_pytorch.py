import numpy as np
import torch
import pyaudio

from mfcc_extractor_lib import hard_limiter, extract_features_live

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK_DURATION = 0.025
CHUNK_SIZE = int(RATE * CHUNK_DURATION)

p = pyaudio.PyAudio()


def process_live(model, device, temporal_shift_windows=6):

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Listening... Press Ctrl+C to stop.")
    try:
        buffer = np.zeros((0,), dtype=np.float32)
        predictions_buffer = []

        while True:
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_buffer = np.frombuffer(audio_chunk, dtype=np.float32)
            buffer = np.concatenate([buffer, audio_buffer])

            if len(buffer) >= CHUNK_SIZE:
                limited_buffer = hard_limiter(buffer, RATE)
                features = extract_features_live(limited_buffer, RATE)

                input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

                with torch.no_grad():
                    predictions = model(input_tensor.view(1, 1, -1))

                predicted_viseme = torch.argmax(predictions, dim=-1).cpu().numpy()
                predictions_buffer.append(predicted_viseme)

                if len(predictions_buffer) > temporal_shift_windows:
                    shifted_viseme = predictions_buffer.pop(0)
                    print(f"Predicted Viseme: {shifted_viseme}")
                else:
                    print("Gathering context...")

                buffer = np.zeros((0,), dtype=np.float32)

    except KeyboardInterrupt:
        print("Stream stopped")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def run_lipsync():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('lipsync_model_entire.pth', map_location=device)

    model.to(device)

    process_live(model, device)

run_lipsync()
