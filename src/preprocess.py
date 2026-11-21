from tensorflow.keras.preprocessing.image import ImageDataGenerator                                      
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50                              
from tensorflow.keras import layers, models                                                                 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = "train"
VALID_DIR = "valid"
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical")
valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
print("Preprocessing completed")
print("Number of classes:", train_generator.num_classes)
print("Class labels:", train_generator.class_indices)
