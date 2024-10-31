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


def remove_single_frame_visemes(current_viseme, previous_viseme, viseme_duration):
    """
    Enforce a minimum duration of two frames for each viseme.

    Parameters:
        current_viseme (int): The currently predicted viseme.
        previous_viseme (int): The previous viseme that was displayed.
        viseme_duration (int): Duration for which the current viseme has been displayed.

    Returns:
        (int, int): A tuple containing the selected viseme and updated viseme duration.
    """
    if current_viseme != previous_viseme:
        if viseme_duration < 2:
            return previous_viseme, viseme_duration + 1
        else:
            return current_viseme, 1
    else:
        return current_viseme, viseme_duration + 1


def process_live(model, device, temporal_shift_windows=6):

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Listening... Press Ctrl+C to stop.")
    try:
        buffer = np.zeros((0,), dtype=np.float32)
        predictions_buffer = deque(maxlen=temporal_shift_windows + 3)
        previous_viseme = None
        viseme_duration = 0

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

                    filtered_viseme = filter_predictions(predictions_buffer)

                    if previous_viseme is None:
                        previous_viseme = filtered_viseme

                    final_viseme, viseme_duration = remove_single_frame_visemes(filtered_viseme, previous_viseme,
                                                                                viseme_duration)
                    previous_viseme = final_viseme

                    print(f"predicted Viseme: {final_viseme}")
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
