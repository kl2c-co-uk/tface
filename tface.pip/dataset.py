import os
import requests

def ensure_directory_exists(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
def download_file(url, save_path):
	
	ensure_directory_exists(save_path)

	if os.path.exists(save_path):
			print(f"The file '{save_path}' already exists. Skipping download.")
			return

	print('this could be loonnngggg .....')
	
	response = requests.get(url, stream=True)
	with open(save_path, 'wb') as f:
			for chunk in response.iter_content(chunk_size=8192):
					if chunk:
							f.write(chunk)
	
	print(f"Downloaded {url} to {save_path}")

##
# wider! http://shuoyang1213.me/WIDERFACE/
# if this doesn't work, visit the website
download_file(
	'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
	'target/wider.training.zip'
)

download_file(
	'https://drive.usercontent.google.com/download?id=1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T&export=download&authuser=0&confirm=t&uuid=7afbbdc2-cbaf-4d4a-8998-16296c6d7ccd&at=APZUnTVUADxFyK6lmy5VgFTEYfUy%3A1719227735059',
	'target/wider.testing.zip'
)

download_file(
	'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=8afa3062-ddbc-44e5-83fd-c4e1e2965513&at=APZUnTUX4c1Le0kpmfMNJ6i3cIJh%3A1719227725353',
	'target/wider.validation.zip'
)

download_file(
	'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip',
	'target/wider.annotations.zip'
)

def safe_write(path):
	ensure_directory_exists(path)
	return open(path, 'w')

import zipfile
import os


# extract the images

# extract the labels
with zipfile.ZipFile('target/wider.annotations.zip', 'r').open('wider_face_split/wider_face_val_bbx_gt.txt') as file_in_zip:
	jpeg = 'target/dataset/validation/images/'
	into = 'target/dataset/validation/annotations/'
	data = iter(file_in_zip.read().decode('utf-8').splitlines())
	def line():
		return next(data)
	print('')
	while data:
		try:
			name = line()
			count = int(line())
			print(f"==== {name}")
			print(f"has {count} faces")

			if not name.endswith('.jpg'):
				raise Exception(f"the file `{name}` does not end with .jpg")

			print('TODO; get height')
			width = ' ?width? '
			height = ' ?height? '

			with safe_write(into + name[:-3] + 'xml') as xml:
				xml.write(f'<annotation>\n')
				xml.write(f'\t<filename>{name}</filename>\n')
				xml.write(f'\t<size>\n')
				xml.write(f'\t\t<width>?width?</width>\n')
				xml.write(f'\t\t<height>?height?</height>\n')
				xml.write(f'\t</size>\n')

				for i in range(count):
					text = line().split(' ')
					x, y, w, h, *_ = text


					xml.write(f'\t<object>\n')
					xml.write(f'\t\t<name>face</name>\n')
					xml.write(f'\t\t<bndbox>\n')
					xml.write(f'\t\t\t<xmin>100</xmin>\n')
					xml.write(f'\t\t\t<ymin>120</ymin>\n')
					xml.write(f'\t\t\t<xmax>250</xmax>\n')
					xml.write(f'\t\t\t<ymax>350</ymax>\n')
					xml.write(f'\t\t</bndbox>\n')
					xml.write(f'\t</object>\n')

				xml.write(f'</annotation>\n')
			print('')
		except StopIteration:
			data = None
			break


raise Exception('do the training set')



print("data set ready - okie dokee")


# <annotation>
#     <filename>image1.jpg</filename>
#     <size>
#         <width>640</width>
#         <height>480</height>
#     </size>
#     <object>
#         <name>person</name>
#         <bndbox>
#             <xmin>100</xmin>
#             <ymin>120</ymin>
#             <xmax>250</xmax>
#             <ymax>350</ymax>
#         </bndbox>
#     </object>
#     <!-- Additional objects if present -->
# </annotation>



