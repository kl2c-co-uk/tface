

> 2024-07-04; this needs to be updated ... sorry


conda create -c conda-forge --prefix ./target/tface-train python=3.10 cudatoolkit=11 cudnn=8 && conda activate ./target/tface-train &&  pip install -r requirements-training.txt




i need a "u-net?" some sort of "new net" i guess







## Setup

### MiniConda

Train with your GPU on Windows!

> this is the difference between over 70 hours and less than 70 minutes of time training

0. install conda
	- https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
1. create the conda environment with "the things"
	- `λ conda create -c conda-forge --prefix ./target/tface-train python=3.10 cudatoolkit=11 cudnn=8`
2. activate that conda environment
	- `λ conda activate ./target/tface-train`
5. setup the pips packages
	- `(tface-train) λ pip install -r requirements-training.txt`

6. run the gpu-check
	- `(tface-train) λ python gpucheck.py`
7. train!
	- `(base) λ python train.py`
7. freeze it
	- `(base) λ python freeze.py`
	- this will create `target/face_detector.tflite`

last three can be done ...

`(tface-train) λ python train.py && python freeze.py && shutdown /s /t 180`

... to work "as one" and shutdown the PC 3 minutes after done.

To abort shutdown, run `shutdown /a` from any shell or console.

### BarraCUDA


swtich to a normla venv (maybe?) since we can't share the training containter


1. create a venv
	- `λ python -m venv target/conversion`
2. activate it
	- `λ "target/conversion/Scripts/activate.bat"`
3. updagrade pip
	- `(conversion) λ python -m pip install --upgrade pip`
4. install tflow eat al
	- `(conversion) λ pip install tensorflow tf2onnx`
5. covnert it
	- `(conversion) λ python -m tf2onnx.convert --tflite target/face_detector.tflite --output ../workspace-tface.unity/Assets/Scenes/face_detector.onnx --opset 13`
6. check on it in the unity project
	- the `.onnx.meta` file and setting may get dropped as i'm ignoring the `.onnx` file
		- ... so ... you *might* have to find the `NULLPOINTER` and drag that asset back over it
	- if the demo shows two spinning cubes you're fine

### Status

- need a WebTexture in the example

-----

> this document was written in a sleep deprived kandor - the author does not predict that this project/subproject will last due to the author's inexperience with the subject matter

original chat;
	https://chatgpt.com/c/5a811017-3ebd-4546-a84e-444a031bd5c0


freeze chat;
	https://chatgpt.com/c/b91f432e-ae85-4a4e-82ed-ed615e2e8057

## dataset

http://shuoyang1213.me/WIDERFACE/

which i then unpack/cache to a layout that the tensorflow stuff can om nom nom easily

(i did this with venv but it seems fine with MiniConda3 as well)



##

Anime Faces (sorry)
- https://github.com/nagadomi/lbpcascade_animeface
- http://www.manga109.org/en/download.html
- http://www.manga109.org/en/download_s.html

https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b

https://github.com/leomaurodesenv/game-datasets?tab=readme-ov-file#dataset



mirrot



git@github.com:g-pechorin/tface.git
