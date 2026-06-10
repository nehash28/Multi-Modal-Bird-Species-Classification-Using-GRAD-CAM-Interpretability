import tensorflow as tf                
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import json

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
TRAIN_DIR = "audio_train_spectrogram"
VALID_DIR = "audio_valid_spectrogram"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"Number of classes:   {NUM_CLASSES}")
print(f"Training samples:    {train_generator.samples}")
print(f"Validation samples:  {valid_generator.samples}")

# ── SAVE CLASS INDICES ─────────────────────────────────────────────────
with open("audio_class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("✅ Class indices saved to audio_class_indices.json")

# ── CLASS WEIGHTS ──────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"✅ Class weights computed for {len(class_weight_dict)} classes")

# ── MODEL ──────────────────────────────────────────────────────────────
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation="softmax")
], name="Audio_Spectrogram_ResNet50")

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── CALLBACKS ──────────────────────────────────────────────────────────
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "audio_model_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# ── PHASE 1: FROZEN BASE ───────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Training with frozen ResNet50 base")
print("="*60)

history_phase1 = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# ── PHASE 2: FINE-TUNING ───────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2: Fine-tuning - Unfreezing top layers")
print("="*60)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_phase2 = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# ── SAVE MODEL ─────────────────────────────────────────────────────────
model.save("audio_model_final.h5")
print("\n✅ Model saved as 'audio_model_final.h5'")

# ── FINAL EVALUATION ───────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

train_loss, train_acc = model.evaluate(train_generator, verbose=0)
val_loss, val_acc     = model.evaluate(valid_generator, verbose=0)

print(f"Training Accuracy:   {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print("="*60)

# ── PLOT ───────────────────────────────────────────────────────────────
all_accuracy     = history_phase1.history["accuracy"]     + history_phase2.history["accuracy"]
all_val_accuracy = history_phase1.history["val_accuracy"] + history_phase2.history["val_accuracy"]
all_loss         = history_phase1.history["loss"]         + history_phase2.history["loss"]
all_val_loss     = history_phase1.history["val_loss"]     + history_phase2.history["val_loss"]

phase1_end = len(history_phase1.history["accuracy"])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(all_accuracy,     label="Training Accuracy")
plt.plot(all_val_accuracy, label="Validation Accuracy")
plt.axvline(x=phase1_end, color='r', linestyle='--', label='Fine-tuning starts')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(all_loss,     label="Training Loss")
plt.plot(all_val_loss, label="Validation Loss")
plt.axvline(x=phase1_end, color='r', linestyle='--', label='Fine-tuning starts')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("audio_training_history.png", dpi=300)
plt.show()

print("\n✅ Training plots saved as 'audio_training_history.png'")
print("\n🎉 Audio model training complete!")