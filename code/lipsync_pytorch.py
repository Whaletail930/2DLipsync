import numpy as np
import torch
import pyaudio

from collections import deque

from mfcc_extractor_lib import hard_limiter, extract_features_live

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK_DURATION = 0.025
CHUNK_SIZE = int(RATE * CHUNK_DURATION)

p = pyaudio.PyAudio()


def filter_predictions(predictions, window_size=3):
    """
    Filter noisy predictions by applying a stability check on viseme transitions.

    Parameters:
        predictions (deque): A deque holding the recent viseme predictions.
        window_size (int): Number of predictions to look ahead for stability check (default: 3).

    Returns:
        int: The filtered viseme prediction.
    """
    if len(predictions) < window_size + 1:
        return predictions[-1]

    if predictions[-1] != predictions[-2]:
        if all(pred == predictions[-1] for pred in list(predictions)[-window_size:]):
            return predictions[-1]
        else:
            return predictions[-2]
    else:
        return predictions[-1]


def process_live(model, device, d=6):

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Listening... Press Ctrl+C to stop.")
    try:
        buffer = np.zeros((0,), dtype=np.float32)
        predictions_buffer = deque(maxlen=d + 3)

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

                # Temporal shift: get the prediction with delay `d`
                if len(predictions_buffer) > d:
                    shifted_viseme = predictions_buffer[-(d + 1)]  # Prediction d steps in the past
                    # Apply noise filtering
                    filtered_viseme = filter_predictions(predictions_buffer)
                    print(f"Filtered Viseme: {filtered_viseme}")
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
