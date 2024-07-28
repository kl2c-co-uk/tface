"""extracts the WIDER to be yolo5 then runs training

"""

import os, shutil

import datasource.config as config
from datasource import Cache, md5, ZipWalk, ensure_directory_exists
import subprocess

import torch
assert(torch.cuda.is_available())

def main(args):

	os.makedirs('target/yolo-dataset/', exist_ok=True)

	cache = Cache('target/')

	##
	# build the datasets
	if args.extract:
		yolo5wider(cache, 'train',
			'wider_face_train_bbx_gt.txt',
			'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
		)

		yolo5wider(cache, 'val',
			'wider_face_val_bbx_gt.txt',
			'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=8afa3062-ddbc-44e5-83fd-c4e1e2965513&at=APZUnTUX4c1Le0kpmfMNJ6i3cIJh%3A1719227725353',
		)

	if args.clone:
		git = yolo5clone()

		if args.train:
			train(git)

		if args.export:
			export(git)

	print('i think that is it')

def yolo5clone():
	##
	# checkout the yolo5 project
	git = os.path.abspath("target/yolov5")
	if not os.path.isdir(git):
		result = subprocess.run(
			["git", "clone", "https://github.com/ultralytics/yolov5.git", git],
			# capture_output=True,
			# text=True
		)

		# Check for errors
		if result.returncode == 0:
			print("YOLOv5 project cloned successfully.")
		else:
			print("Error cloning YOLOv5 project:", result.stderr)
	return git

def train(git):
	##
	# write the/a yaml
	yaml = os.path.abspath('target/yolo5.yaml')
	with open(yaml, 'w') as file:
		import textwrap
		file.write(
			textwrap.dedent(f"""\
				train: {os.path.abspath('target/yolo-dataset/images/train')}
				val: {os.path.abspath('target/yolo-dataset/images/val')}
				nc: 1  # number of classes
				names:
				- face
			""")
		)


	##
	# train it?
	print('launching the training ...')
	result = subprocess.run([
			'python', 'train.py',
			'--img', str(config.INPUT_SIZE),
			'--batch', str(config.BATCH_SIZE),
			'--epochs', str(config.EPOCHS),
			'--data', yaml,
			'--weights','yolov5s.pt'
		],
		cwd=git,
		# capture_output=True,
		# text=True
	)

	# Check for errors
	if result.returncode == 0:
		print("... yolo5 trained successfully.")
	else:
		print("Error training YOLOv5 project:", result.stderr)

def export(git):
	##
	# export the model
	print('exporting ...')
	experiments = os.listdir(git + '/runs/train/')
	experiments.sort()
	weights = (git + '/runs/train/' + experiments[-1] + '/weights/best.pt')

	assert os.path.isfile(weights), f"Trained weights file not found: {weights}"

	result = subprocess.run([
			"python", "export.py",
			"--weights", str(weights),
			"--img-size", str(config.INPUT_SIZE),
			"--batch-size", str(config.EXPORT_BATCH_SIZE),
			"--device", "cpu",  # or "cuda" for GPU
								# ... pal doesn't think this matters ...

			'--opset', '9',	# unix's barracuda is OLD so use the old version

			"--include", "onnx"  # specify the format to export
		],
		cwd=git,
		# capture_output=True,
		# text=True
	)
	# Check for errors
	if result.returncode == 0:
		print("Model exported successfully to ONNX format.")
		# print(result.stdout)
	else:
		print("Error during model export:", result.stderr)
	weights = weights[:-2]+'onnx'
	
	##
	# simply it into the unity project
	import onnx
	from onnxsim import simplify

	# Load your ONNX model
	onnx_model = onnx.load(weights)
	print(f'onnx_model.opset_import[0].version = {onnx_model.opset_import[0].version}')
	assert 9 == onnx_model.opset_import[0].version

	# Simplify the model
	model_simp, check = simplify(onnx_model)

	# Save the simplified model
	out = os.path.abspath('../workspace-tface.unity/Assets/Scenes/yoloface.onnx')
	onnx.save(model_simp, out)

def yolo5wider(cache, group, txt, url):
	# download the annotations file
	annotations = cache.download(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
	)

	# download the images file
	images = cache.download(url)

	point_count = 0

	for point in wider(annotations, txt):
		if config.LIMIT > 0 and point_count >= config.LIMIT:
			break
		else:
			point_count += 1
			
			path, faces = point

			fKey = md5(path)

			jpg = f'target/yolo-dataset/images/{group}/{fKey}.jpg'
			txt = f'target/yolo-dataset/labels/{group}/{fKey}.txt'

			if os.path.isfile(jpg) and os.path.isfile(txt):
				continue

			ensure_directory_exists(jpg)
			ensure_directory_exists(txt)

			for data in ZipWalk(images).read(path):
				import cv2
				import numpy as np 

				iw, ih, _ = cv2.imdecode(
					np.frombuffer(data, dtype=np.uint8),
					cv2.IMREAD_COLOR).shape
				
				dw = 1.0 / float(iw)
				dh = 1.0 / float(ih)

				labels = []
				skipped = 0
				for face in faces:
					try:
						x, y, w, h = face

						assert 0 <= x
						assert (x+w) < iw, f'for `{group}` >{path}< iw = {iw} and x={x}, w={w}'
						assert 0 <= y
						assert (y+h) < ih

						x = (x + (w/2)) * dw
						w *= dw

						y = (y + (h/2)) * dh
						h *= dh

						labels.append(f'0 {x} {y} {w} {h}')
					except AssertionError as e:
						skipped += 1
				if 0 != skipped:
					
					print(f'skipped {group} / {fKey}\n\tnamed {path}\n\tbecause {skipped} out of {len(faces)} faces were out of bounds\n')
					point_count -= 1
				else:
					# write the bytes
					with open(jpg, 'wb') as file:
						file.write(data)
					# write the labels
					with open(txt, 'w') as file:
						for label in labels:
							file.write(label)
							file.write('\n')

	print(f'prepared dataset >{group}< of {point_count} points (or more!)')

def wider(annotations, txt):
	"""given an annotateins file, this emits `(jpeg, [(x,y,w,h)])` data points"""
	
	for lines in ZipWalk(annotations).text(txt):
		while lines.more():
			jpeg_path = lines.take()
			face_count = int(lines.take())
			# read entries
			faces = []
			if 0 == face_count:
				# for extra weirdness; entries (or The One Entry) with no faces have a line with garbage data
				blank = lines.take().strip()
				if '0 0 0 0 0 0 0 0 0 0 '.strip() != blank:
					raise Exception('empty entry had a funky line!')
			else:
				while len(faces) < face_count:
					
					# grab the first four and convert them to ints
					x, y, w, h = map(int, lines.take().split(' ')[:4])
					
					if 0 == w or 0 == h:
						face_count -= 1
					else:
						assert w > 0
						assert h > 0

						faces.append(
							(x, y, w, h)
						)

			# just check that face count didn't get weird
			assert 0 <= face_count
			
			yield (jpeg_path, faces)

if '__main__' == __name__:
	import argparse
	parser = argparse.ArgumentParser(description="YOLOv5 Face Detector (maybe)")
	parser.add_argument('-extract', action='store_true', help='only do the extraction step')
	args = parser.parse_args()

	class Blurb:
		def __init__(self, **kwargs):
			for key, value in kwargs.items():
				setattr(self, key, value)

	if args.extract:
		args = Blurb(
			extract = True,
			clone = False,
			train = False,
			export = False
		)
	else:
		args = Blurb(
			extract = True,
			clone = True,
			train = True,
			export = True
		)
	main(args)
