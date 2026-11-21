import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
IMAGE_MODEL_PATH = "resnet50_birds25.h5"
AUDIO_MODEL_PATH = "audio_model_best.h5"
IMAGE_CLASS_JSON = "class_indices.json"
AUDIO_CLASS_JSON = "audio_class_indices.json"

IMAGE_VAL_DIR    = "valid"                   
AUDIO_VAL_DIR    = "audio_valid_spectrogram" 

IMAGE_SIZE       = (224, 224)


print("Loading models...")
image_model = load_model(IMAGE_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = image_model(dummy, training=False)
_ = audio_model(dummy, training=False)
print("Models loaded successfully")

with open(IMAGE_CLASS_JSON) as f:
    image_class_indices = json.load(f)
with open(AUDIO_CLASS_JSON) as f:
    audio_class_indices = json.load(f)

image_idx_to_class = {v: k for k, v in image_class_indices.items()}
audio_idx_to_class = {v: k for k, v in audio_class_indices.items()}

print(f" Image classes : {len(image_idx_to_class)}")
print(f" Audio classes : {len(audio_idx_to_class)}")


def preprocess_image(img_path):
    img       = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def preprocess_spectrogram(spec_path):
    img       = image.load_img(spec_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
def confidence_based_fusion(image_probs, audio_probs):
    conf_image = float(np.max(image_probs))
    conf_audio = float(np.max(audio_probs))
    total      = conf_image + conf_audio
    w_image    = conf_image / total
    w_audio    = conf_audio / total

    all_classes  = sorted(set(image_idx_to_class.values()))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    image_unified = np.zeros(len(all_classes))
    for idx, prob in enumerate(image_probs[0]):
        cls = image_idx_to_class.get(idx)
        if cls and cls in class_to_idx:
            image_unified[class_to_idx[cls]] = prob

    audio_unified = np.zeros(len(all_classes))
    for idx, prob in enumerate(audio_probs[0]):
        cls = audio_idx_to_class.get(idx)
        if cls and cls in class_to_idx:
            audio_unified[class_to_idx[cls]] = prob

    fused_probs = w_image * image_unified + w_audio * audio_unified
    top_idx     = np.argmax(fused_probs)
    return all_classes[top_idx], float(fused_probs[top_idx])
def evaluate_full_dataset():

    species_list = sorted(os.listdir(IMAGE_VAL_DIR))
    print(f"\nFound {len(species_list)} species folders\n")

    total         = 0
    img_correct   = 0
    aud_correct   = 0
    fused_correct = 0
    per_species_results = {}

    for species in species_list:
        img_species_dir = os.path.join(IMAGE_VAL_DIR, species)
        aud_species_dir = os.path.join(AUDIO_VAL_DIR, species)

        if not os.path.isdir(img_species_dir):
            continue
        if not os.path.isdir(aud_species_dir):
            print(f"No spectrogram folder for {species}, skipping...")
            continue
        img_files = sorted([
            f for f in os.listdir(img_species_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        aud_files = sorted([
            f for f in os.listdir(aud_species_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        pairs = min(len(img_files), len(aud_files))

        if pairs == 0:
            print(f"No paired samples for {species}, skipping...")
            continue

        print(f"{species:<30} → {pairs} samples")

        sp_total   = 0
        sp_img_ok  = 0
        sp_aud_ok  = 0
        sp_fuse_ok = 0

        for j in range(pairs):
            img_path = os.path.join(img_species_dir, img_files[j])
            aud_path = os.path.join(aud_species_dir, aud_files[j])

            try:
                img_input  = preprocess_image(img_path)
                aud_input  = preprocess_spectrogram(aud_path)  

                image_probs = image_model.predict(img_input,  verbose=0)
                audio_probs = audio_model.predict(aud_input,  verbose=0)

                img_pred   = image_idx_to_class[np.argmax(image_probs)]
                audio_pred = audio_idx_to_class[np.argmax(audio_probs)]
                fused_pred, _ = confidence_based_fusion(image_probs, audio_probs)

                if img_pred   == species: sp_img_ok  += 1
                if audio_pred == species: sp_aud_ok  += 1
                if fused_pred == species: sp_fuse_ok += 1
                sp_total += 1

            except Exception as e:
                print(f" Error: {e}")
                continue

        total         += sp_total
        img_correct   += sp_img_ok
        aud_correct   += sp_aud_ok
        fused_correct += sp_fuse_ok

        per_species_results[species] = {
            "total"      : sp_total,
            "image_acc"  : sp_img_ok  / sp_total * 100 if sp_total > 0 else 0,
            "audio_acc"  : sp_aud_ok  / sp_total * 100 if sp_total > 0 else 0,
            "fusion_acc" : sp_fuse_ok / sp_total * 100 if sp_total > 0 else 0,
        }
    if total == 0:
        print("\n No samples found! Check folder paths and file types.")
        return
    print("\n\n" + "=" * 72)
    print("                    PER SPECIES ACCURACY")
    print("=" * 72)
    print(f"{'Species':<28} {'Samples':<10} {'Image':>10} {'Audio':>10} {'Fusion':>10}")
    print("-" * 72)
    for sp, res in per_species_results.items():
        print(f"{sp:<28} {res['total']:<10} "
              f"{res['image_acc']:>9.2f}% "
              f"{res['audio_acc']:>9.2f}% "
              f"{res['fusion_acc']:>9.2f}%")
    img_acc   = img_correct   / total * 100
    aud_acc   = aud_correct   / total * 100
    fused_acc = fused_correct / total * 100

    print("\n" + "=" * 72)
    print("                  OVERALL ACCURACY SUMMARY")
    print("=" * 72)
    print(f"  Total Samples Tested  : {total}")
    print(f"  Image  Accuracy       : {img_correct}/{total}  →  {img_acc:.2f}%")
    print(f"  Audio  Accuracy       : {aud_correct}/{total}  →  {aud_acc:.2f}%")
    print(f"  Fusion Accuracy       : {fused_correct}/{total}  →  {fused_acc:.2f}%")
    print("=" * 72)

    winner = max(
        ("Image",  img_acc),
        ("Audio",  aud_acc),
        ("Fusion", fused_acc),
        key=lambda x: x[1]
    )
    print(f"\n Best Model → {winner[0]} ({winner[1]:.2f}%)")
    print("=" * 72)

if __name__ == "__main__":
    evaluate_full_dataset()