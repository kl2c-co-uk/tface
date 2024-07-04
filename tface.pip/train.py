
# Paths to training and validation directories
from dataset import dataset_main
dataset_norms =  dataset_main()

print('2024-07-04; stuff beyon here hasnt been tested')
print(f"dataset_norms = `{dataset_norms}`")
raise '???'

train_image_dir			= dataset_norms+  '/train/images'
train_mask_dir			= dataset_norms+  '/train/masks'
validation_image_dir	= dataset_norms+  '/validation/images'
validation_mask_dir		= dataset_norms+  '/validation/masks'

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

# Define the U-Net model
def unet_model(input_size=(224, 224, 3)):
    inputs = Input(input_size)
    
    # Encoder: Using a pre-trained ResNet50 as the encoder
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)
    
    # Collect encoder outputs for skip connections
    skip_connections = [base_model.get_layer(name).output for name in [
        "conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"
    ]]
    encoder_output = base_model.get_layer("conv5_block3_out").output

    # Decoder
    def decoder_block(input_tensor, skip_tensor, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
        x = concatenate([x, skip_tensor])
        x = Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
        return x

    d1 = decoder_block(encoder_output, skip_connections[3], 512)
    d2 = decoder_block(d1, skip_connections[2], 256)
    d3 = decoder_block(d2, skip_connections[1], 128)
    d4 = decoder_block(d3, skip_connections[0], 64)
    
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(d4)
    
    model = Model(inputs, outputs)
    return model

# Custom data generator for images and masks
def image_mask_generator(image_dir, mask_dir, batch_size, target_size):
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)
    
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        seed=1)
    
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=1)
    
    while True:
        img_batch = image_generator.next()
        mask_batch = mask_generator.next()
        yield img_batch, mask_batch















# Generator parameters
batch_size = 16
target_size = (224, 224)

# Create generators
train_generator = image_mask_generator(train_image_dir, train_mask_dir, batch_size, target_size)
validation_generator = image_mask_generator(validation_image_dir, validation_mask_dir, batch_size, target_size)

# Define the model
model = unet_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Calculate the steps per epoch
train_steps = len(os.listdir(train_image_dir)) * batch_size
validation_steps = len(os.listdir(validation_image_dir)) * batch_size

# Train the model
model.fit(train_generator, steps_per_epoch=train_steps, validation_data=validation_generator, validation_steps=validation_steps, epochs=10)

# Save the model
model.save('target/face_detector.keras')

print('Model trained and saved!')
