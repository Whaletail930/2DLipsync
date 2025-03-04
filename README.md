# Real-Time Lip Sync for Live 2D Animation

## Overview

This project is an **open-source implementation** of a **real-time lip sync system** for live 2D animation, inspired by the paper *Real-Time Lip Sync for Live 2D Animation* by **Deepali Aneja & Wilmot Li (2019)**. The implementation utilizes a **Long Short-Term Memory (LSTM)** deep learning model trained on custom datasets to generate visemes from live audio input.

## Features

- **Live Audio Input**: Captures audio in real-time and extracts features using **Librosa**.
- **LSTM Model**: Processes audio features and predicts viseme sequences.
- **Frame Rate Adaptation**: Outputs predictions at **100Hz** and downsamples to **24 FPS** for smooth animation.
- **Modular Design**: Easy integration into different applications.
- **Noise Filtering & Silence Detection**: Reduces unwanted noise and correctly handles silent moments.
- **Dataset Generation**: Uses the **TIMIT dataset** for speech data and **Rhubarb Lip Sync** for automatic viseme labeling.
- **Tkinter-Based Animation GUI**: A simple **Python GUI** displays animated mouth movements based on the predicted visemes.
- **Scalability**: Can be extended with **different datasets**, **multi-language support**, or **3D animation integration**.

## Installation

### Requirements

- Python 3.8+
- **PyTorch** (for LSTM model training and inference)
- **Librosa** (for audio feature extraction)
- **Tkinter** (for GUI animation)
- **PyAudio** (for live microphone input processing)
- **NumPy, Pandas** (for data handling)

### Setup

**Download the TIMIT Dataset** (Optional, can be other dataset as well):

   - Obtain the dataset from [LDC](https://catalog.ldc.upenn.edu/LDC93S1)
   - Place `.wav` files in DATA folder

## References

- *Deepali Aneja & Wilmot Li (2019): Real-Time Lip Sync for Live 2D Animation.*
- TIMIT Acoustic-Phonetic Continuous Speech Corpus (LDC93S1).
- Rhubarb Lip Sync ([https://github.com/DanielSWolf/rhubarb-lip-sync](https://github.com/DanielSWolf/rhubarb-lip-sync)).

