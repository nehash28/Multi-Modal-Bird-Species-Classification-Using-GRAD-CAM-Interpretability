from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

TRAIN_DIR = "train" 

datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical"
)

with open("class_indices.json", "w") as f:
    json.dump(generator.class_indices, f)

print("✅ Saved class_indices.json")
print(generator.class_indices)