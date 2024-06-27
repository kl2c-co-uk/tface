
# import tensorflow as tf

# # List all available devices
# devices = tf.config.list_physical_devices()
# print("Available devices:")
# for device in devices:
#     print('')
#     print('=====')
#     print(device)

# # Check TensorFlow version and GPU availability
# print('')
# print('=====')
# print("TensorFlow version:", tf.__version__)
# print("GPU available:", tf.test.is_gpu_available())




import tensorflow as tf

ver = tf.__version__
bwc = tf.test.is_built_with_cuda()
num = len(tf.config.experimental.list_physical_devices('GPU'))

print("=" * 41)

print("TensorFlow version:", ver)
print("Built with CUDA:", bwc)
print("Num GPUs Available:", num)


print("-" * 41)

import sys
import os

for path in os.environ['PATH'].split(';'):
	if os.path.isdir(path):
		for file in os.listdir(path):
			file = file.lower()
			if file.endswith('.dll') and 'cuda' in file:
				if path:
					print("\t", path)
					path = False
				print("\t\t", file)


