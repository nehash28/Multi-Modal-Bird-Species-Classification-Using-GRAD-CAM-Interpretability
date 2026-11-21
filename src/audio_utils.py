"""Shared audio -> mel-spectrogram code.

Both the dataset conversion (audio_to_spectrogram.py) and inference (fusion.py)
import from here, so the audio model always sees spectrograms generated with
exactly the same settings it was trained on.
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050
MAX_DURATION = 5  # seconds


def load_audio(file_path):
    """Load a .wav, trim silence, then pad/cut to MAX_DURATION seconds."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Remove silence
    y, _ = librosa.effects.trim(y)

    # Fix duration (pad or cut to 5 sec)
    target_len = SAMPLE_RATE * MAX_DURATION
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y, sr


def save_spectrogram_png(y, sr, save_path):
    """Render a mel-spectrogram of `y` to a PNG (same settings as training)."""
    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(2.5, 2.5))
    librosa.display.specshow(spectrogram_db, sr=sr, cmap='viridis')
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def audio_file_to_spectrogram_png(audio_path, save_path):
    """.wav file -> spectrogram PNG. Returns save_path."""
    y, sr = load_audio(audio_path)
    save_spectrogram_png(y, sr, save_path)
    return save_path
