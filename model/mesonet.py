# mesonet.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense
)
from tensorflow.keras.optimizers import Adam

def build_meso(input_shape=(256, 256, 3)):
    """
    Builds an optimized MesoNet model for deepfake detection.
    
    Parameters:
        input_shape (tuple): Shape of the input image (default is (256, 256, 3)).
    
    Returns:
        model (keras.Model): Compiled Keras model ready for training.
    """
    
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(8, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = Conv2D(8, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3
    x = Conv2D(16, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 4
    x = Conv2D(16, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    # Fully Connected Layers
  
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)  x = Flatten()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Define and compile model
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
