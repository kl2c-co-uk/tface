import os
import requests

def download_file(url, save_path):
	def ensure_directory_exists(file_path):
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
    
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




# # Example usage
# url = 'https://example.com/archive.zip'  # Replace with the actual URL of the zip file
# save_path = 'downloaded_archive.zip'     # Specify where to save the downloaded file

# download_file(url, save_path)
# print(f"Downloaded {url} to {save_path}")


