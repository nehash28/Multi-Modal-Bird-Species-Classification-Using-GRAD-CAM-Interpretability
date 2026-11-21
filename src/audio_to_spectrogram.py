import os
import matplotlib.pyplot as plt

from audio_utils import audio_file_to_spectrogram_png

SOURCE_TRAIN = "audio_train"
SOURCE_VALID = "audio_valid"

OUTPUT_TRAIN = "audio_train_spectrogram"
OUTPUT_VALID = "audio_valid_spectrogram"


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
                audio_file_to_spectrogram_png(file_path, save_path)

                converted += 1
                print(f"  ✅ [{i+1}/{len(wav_files)}] {file}")

            except Exception as e:
                print(f"  ❌ {file}: {e}")
                plt.close()

        print(f"✅ Done: {species} — {converted} converted, {skipped} skipped\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_VALID, exist_ok=True)

    convert_folder(SOURCE_TRAIN, OUTPUT_TRAIN)
    convert_folder(SOURCE_VALID, OUTPUT_VALID)

    print("🎉 All spectrograms generated successfully!")
