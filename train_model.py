import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import Adam

# ------------------------------
# Paths
# ------------------------------
DATA_DIR = "dataset"  # root folder
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_PATH = "model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# ------------------------------
# Parameters
# ------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# ------------------------------
# Data generators
# ------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # multi-class (3 classes)
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ------------------------------
# Save class indices
# ------------------------------
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_generator.class_indices, f)
print("Saved class indices:", train_generator.class_indices)

# ------------------------------
# Build model
# ------------------------------
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(3, activation="softmax")(x)  # 3 classes

model = Model(inputs=base_model.input, outputs=output)

# ------------------------------
# Compile
# ------------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------
# Train
# ------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ------------------------------
# Save model
# ------------------------------
model.save(MODEL_PATH)
print("Model saved as", MODEL_PATH)
