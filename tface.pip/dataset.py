import os
import requests

def ensure_directory_exists(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory
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
import cv2

def throw(message):
	print('\n')
	raise Exception(f"{message}\n\n")

# the validation set
root = 'target/dataset/validation/'
labels = 'wider_face_split/wider_face_val_bbx_gt.txt'

# extract the images
with zipfile.ZipFile('target/wider.validation.zip', 'r') as full:
	for file_info in full.infolist():
		if file_info.filename.endswith('.jpg'):
			name = file_info.filename 
			name = root+name[name.find('images/'):]
			if not os.path.isfile(name):
				ensure_directory_exists(name)
				with open(name, 'wb') as file:
					file.write(full.read(file_info))

# extract the labels
with zipfile.ZipFile('target/wider.annotations.zip', 'r').open(labels) as file_in_zip:
	
	##
	# finky recreation of an iterator
	data = {
		'next': 0,
		'full': file_in_zip.read().decode('utf-8').splitlines(),
		'more': True
	}
	
	def line(data):
		item = data['full'][data['next']]
		data['next'] += 1
		data['more'] = data['next'] < len(data['full'])
		return item

	
	print('')

	
	def xml_write(xml, string):
		xml.write(string + '\n')

	while data['more']:

		# read the file name for this entry
		name = line(data)
		if not name.endswith('.jpg'):
			raise Exception(f"the file `{name}` does not end with .jpg")

		# open teh label xml file
		xml = root+ 'annotations/' +name[:-3] + 'xml'
		xml = safe_write(xml)

		# load the count of targets
		count = int(line(data))

		# load the image and find its dimensions
		image = root+'images/'+name
		if not os.path.isfile(image):
			print('\n')
			raise Exception(f"the file `{image}` does not exist\n\n")
		image = cv2.imread(image)
		if image is None:
			throw(f'failed to load {name}')
		height, width, channels = image.shape

		# write the header for the record
		xml_write(xml, f'<annotation>')
		xml_write(xml, f'\t<filename>{name}</filename>')
		xml_write(xml, f'\t<size>')
		xml_write(xml, f'\t\t<width>{width}</width>')
		xml_write(xml, f'\t\t<height>{height}</height>')
		xml_write(xml, f'\t</size>')

		# wander through each item in the record
		for i in range(count):
			text = line(data).split(' ')
			x, y, w, h, *_ = text

			xml_write(xml, f'\t<object>')
			xml_write(xml, f'\t\t<name>face</name>')
			xml_write(xml, f'\t\t<bndbox>')
			xml_write(xml, f'\t\t\t<xmin>{x}</xmin>')
			xml_write(xml, f'\t\t\t<ymin>{y}</ymin>')
			xml_write(xml, f'\t\t\t<xmax>{x+w}</xmax>')
			xml_write(xml, f'\t\t\t<ymax>{y+h}</ymax>')
			xml_write(xml, f'\t\t</bndbox>')
			xml_write(xml, f'\t</object>')

		xml_write(xml, f'</annotation>')
		xml.close()


raise Exception('do the training set')



print("data set ready - okie dokee")


# <annotation>
#	 <filename>image1.jpg</filename>
#	 <size>
#		 <width>640</width>
#		 <height>480</height>
#	 </size>
#	 <object>
#		 <name>person</name>
#		 <bndbox>
#			 <xmin>100</xmin>
#			 <ymin>120</ymin>
#			 <xmax>250</xmax>
#			 <ymax>350</ymax>
#		 </bndbox>
#	 </object>
#	 <!-- Additional objects if present -->
# </annotation>



