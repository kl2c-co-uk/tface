
import tensorflow as tf

# touch stuff
tf.keras.models.load_model

print('')

# Load the saved model
model = tf.keras.models.load_model('target/face_detector.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open('target/face_detector.tflite', 'wb') as f:
	f.write(tflite_model)

print("file saved")
