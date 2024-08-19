"""extracts the WIDER to be yolo5 then runs training

"""

import os, shutil

import datasource.config as config
from datasource import Blurb, Cache, md5, ZipWalk, ensure_directory_exists
import subprocess
from  datasource.datapoints import FacePatch, DataPoint
import datasource.datapoints as datapoints
from datasource.datapoints import split_export

import torch
assert(torch.cuda.is_available())

def main(args):

	os.makedirs(f'target/yolo-dataset_{config.LIMIT}/', exist_ok=True)

	cache = Cache('target/')

	##
	# build the datasets
	if args.extract:

		# do the newer cartoon ones
		cartoon_archive = cache.download(
			# training = target/cb67961c4ba344c84b3e5442206436ac
			'https://drive.usercontent.google.com/download?id=1xXpE0qs2lONWKL5dqaFxqlJ_t5-glNpg&export=download&authuser=0&confirm=t&uuid=f6f6beb7-4c3b-40a7-b52d-12c62c2e84fe&at=APZUnTV9QwxtWfOsgjgqW-7icoaM:1723671279280'
		)
		split_export(
			datapoints.greenlist(
				i_cartoon_datapoints(cache),
				cartoon_archive
			),
			8, 1,
			cartoon_archive
		)

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
			train(args, git)

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

def train(args, git):


	##
	# write the/a yaml
	yaml = os.path.abspath('target/yolo5.yaml')
	with open(yaml, 'w') as file:
		import textwrap
		file.write(
			textwrap.dedent(f"""\
				train: {os.path.abspath(f'target/yolo-dataset_{config.LIMIT}/images/train')}
				val: {os.path.abspath(f'target/yolo-dataset_{config.LIMIT}/images/val')}
				nc: 1  # number of classes
				names:
				- face
			""")
		)

	weights = args.weights or 'yolov5s.pt'
	##
	# train it?
	print(f'launching the training from {weights} ...')
	result = subprocess.run([
			'python', 'train.py',
			'--img', str(config.INPUT_SIZE),
			'--batch', str(config.BATCH_SIZE),
			'--epochs', str(config.EPOCHS),
			'--data', yaml,
			'--weights', weights
		],
		cwd=git,
		# capture_output=True,
		# text=True
	)

	# Check for errors
	if result.returncode == 0:
		print("... yolo5 trained successfully.")
	else:
		raise Exception("Error training YOLOv5 project:", result.stderr)

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
	"""fully extract this dataset"""


	# download the annotations file
	annotations = cache.download(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
	)

	# download the images file
	images = cache.download(url)
	print(f"the archive at {url} became {images}")

	# adapt the older format (from July) to work with the newer approach (hey August)
	def adapt():
		for point in wider(annotations, txt):
			
			patches = []
			for l, t, w, h in point[1]:
				r = l + w
				b = t + h
				patches.append(
					FacePatch(ltrb=[l, t, r, b])
				)

			yield DataPoint(
				f'WIDER_{group}/images/{point[0]}',
				patches)
	split_export(
		adapt(), 
		1 if 'train' == group else 0,
		1 if 'val' == group else 0,
		images)

def i_cartoon_datapoints(cache):
	annotations = cache.download(
		# annotations = target/712e3f96290bfc9c1c93a18f16ef40e8
		'https://drive.usercontent.google.com/download?id=15IHSlNBZBZs_hj6B341swc00ha5fpvB7&export=download&authuser=0&confirm=t&uuid=72fd55fe-6a76-4c73-91ee-63de54aa2775&at=APZUnTXR3ogM4tIFCGrcFoaswAor:1723670943322'
	)


	for lines in ZipWalk(annotations).text('personai_icartoonface_dettrain_anno_updatedv1.0.csv'):
		seen = []

		# start a non-datapoint
		last = ''
		data = '?unset?'

		while lines.more():
			line = lines.take()

			# get the line content
			name, l, t, r, b = line.split(',')

			# check if we need to switch datapoints
			if name != last:

				# emit the prior datapoint
				if '' != last:					
					yield DataPoint(
						# we're doing the full path now for my sanity
						path = f'personai_icartoonface_dettrain/icartoonface_dettrain/{last}',
						patches = data
					)
				
				# start a datapoint
				data = []
				last = name

				# check the names. maybe
				if name in seen:
					raise Exception(
						'the items are not grouped as i d expected'
					)
				seen.append(name)
			
			# add the patch tot he datapoint
			data.append(
				FacePatch(ltrb = [l, t, r, b])
			)


		
		# yield the final datapooint
		if '' != last:					
			yield DataPoint(path = last, patches = data)							

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

	parser.add_argument('--weights', type=str, help='Path to the .pt file i should resume/start from')

	args = parser.parse_args()

	##
	# check the resume paramter
	if not args.weights:
		print('training from scratch')
	else:
		# nodemon --ignore target/ yolo5.py --weights "G:/My Drive/kl2c/best.pt"

		if os.path.isfile(args.weights):
			was = args.weights
			args.weights = args.weights.replace('\\','/')
			args.weights = os.path.abspath(args.weights)
			args.weights = args.weights.replace('\\','/')

			print(f"resuming/retraining from `{args.weights}`")
			if args.weights != was:
				print(f"  (... which was {was})")

	if args.extract:
		args = Blurb(
			weights = args.weights,
			extract = True,
			clone = False,
			train = False,
			export = False
		)
	else:
		args = Blurb(
			weights = args.weights,
			extract = True,
			clone = True,
			train = True,
			export = True
		)
	main(args)
