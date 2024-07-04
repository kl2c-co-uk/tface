
print('2024-07-03; this is old and needs to be replaced with heat-mapped based logic based on image data extracted already')


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dataset import throw

# ChatGPT said 32, but, that crashed it. 8 also worked, but, those seems faster.
training_batch_size = 8
training_epochs = 1

input_image_w = 1920
input_image_h = 1080
input_image_scale = 0.2 # ajust it to 1/5th

# originally was 1024, but, the dataset garbage so whatever
layer_mid = 8


input_image_shape = (
	int(input_image_w * input_image_scale),
	int(input_image_h * input_image_scale),
)

# Load and preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory('target/dataset/train', target_size=input_image_shape, batch_size=training_batch_size, subset='training')
validation_generator = train_datagen.flow_from_directory('target/dataset/validation', target_size=input_image_shape, batch_size=training_batch_size, subset='validation')

# Load a pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_image_shape[0], input_image_shape[1], 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(layer_mid, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Assuming binary classification (face/no-face)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=training_epochs)

# Save the model
model.save('target/face_detector.keras')

print('trained you say? okie dokie')
