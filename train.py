import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.mesonet import build_meso  # Updated import
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Paths
train_dir = 'Dataset/Train'
val_dir = 'Dataset/Validation'
model_save_path = 'model/mesonet_trained.h5'

# Image dimensions
img_height, img_width = 256, 256
batch_size = 16
epochs = 10

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True  # good practice for training
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build model
model = build_meso(input_shape=(img_height, img_width, 3))

# Callbacks
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, lr_scheduler]
)
