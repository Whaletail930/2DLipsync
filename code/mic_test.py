import pyaudio
import wave

# Parameters for the audio recording
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono audio
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 1024              # 2^10 samples per frame
RECORD_SECONDS = 10       # Duration of recording in seconds
OUTPUT_FILENAME = "output.wav"  # Output file name

# Create an interface to PortAudio
audio = pyaudio.PyAudio()

# Open a new stream for recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

# Initialize an empty list to store the frames
frames = []

# Loop through stream and append audio chunks to frames list
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate the PortAudio interface
audio.terminate()

# Save the recorded data as a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved as {OUTPUT_FILENAME}")

