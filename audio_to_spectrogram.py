import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

SOURCE_TRAIN = "audio_train"
SOURCE_VALID = "audio_valid"

OUTPUT_TRAIN = "audio_train_spectrogram"
OUTPUT_VALID = "audio_valid_spectrogram"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VALID, exist_ok=True)

MAX_DURATION = 5  
SAMPLE_RATE = 22050

def convert_folder(source_dir, output_dir):
    for species in os.listdir(source_dir):
        species_path = os.path.join(source_dir, species)

        if not os.path.isdir(species_path):
            continue

        output_species_path = os.path.join(output_dir, species)
        os.makedirs(output_species_path, exist_ok=True)

        wav_files = [f for f in os.listdir(species_path) if f.endswith(".wav")]
        converted = 0
        skipped = 0

        for i, file in enumerate(wav_files):
            save_path = os.path.join(output_species_path, file.replace(".wav", ".png"))

            
            if os.path.exists(save_path):
                skipped += 1
                continue

            file_path = os.path.join(species_path, file)

            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Remove silence
                y, _ = librosa.effects.trim(y)

                # Fix duration (pad or cut to 5 sec)
                if len(y) > SAMPLE_RATE * MAX_DURATION:
                    y = y[:SAMPLE_RATE * MAX_DURATION]
                else:
                    y = np.pad(y, (0, SAMPLE_RATE * MAX_DURATION - len(y)))

                # Create Mel Spectrogram
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

                converted += 1
                print(f"  ✅ [{i+1}/{len(wav_files)}] {file}")

            except Exception as e:
                print(f"  ❌ {file}: {e}")
                plt.close()

        print(f"✅ Done: {species} — {converted} converted, {skipped} skipped\n")

convert_folder(SOURCE_TRAIN, OUTPUT_TRAIN)
convert_folder(SOURCE_VALID, OUTPUT_VALID)

print("🎉 All spectrograms generated successfully!")