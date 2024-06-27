

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



setupswtich to a 




2. setup the outer environment;
	- `λ conda install -c conda-forge cudatoolkit=11 cudnn=8`
3. activate the environment
	- `λ conda activate`
4. install python 3.10 in the condan environment
	- `(base) λ conda install python=3.10`
5. setup the pips packages
	- `(base) λ pip install -r requirements.txt`
6. run the gpu-chekc
	- `(base) λ python gpucheck.py`
7. train!
	- `(base) λ python train.py`
7. freeze it
	- `(base) λ python freeze.py`
	- this will create `target/face_detector.tflite`










---------




> this document was written in a sleep deprived kandor - the author does not predict that this project/subproject will last due to the author's inexperience with the subject matter

original chat;
	https://chatgpt.com/c/5a811017-3ebd-4546-a84e-444a031bd5c0


freeze chat;
	https://chatgpt.com/c/b91f432e-ae85-4a4e-82ed-ed615e2e8057

## dataset

http://shuoyang1213.me/WIDERFACE/

which i then unpack/cache to a layout that the tensorflow stuff can om nom nom easily

(i did this with venv but it seems fine with MiniConda3 as well)

## MiniConda3

based on "rtfm" approach i switched to the MiniConda instructions to try and get it to work with GPUs.
embarasingly - conda seems to handle my CUDA concerns for me ... so there ...

1. install conda
	- https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
2. setup the outer environment;
	- `λ conda install -c conda-forge cudatoolkit=11 cudnn=8`
3. activate the environment
	- `λ conda activate`
4. install python 3.10 in the condan environment
	- `(base) λ conda install python=3.10`
5. setup the pips packages
	- `(base) λ pip install -r requirements.txt`
6. run the gpu-chekc
	- `(base) λ python gpucheck.py`
7. train!
	- `(base) λ python train.py`
7. freeze it
	- `(base) λ python freeze.py`
	- this will create `target/face_detector.tflite`

## In Unity

Unity's Baracuda should be sensible to use rather than trying to tape something else into Unity.

So ... I need to convert my model to ONNX.

The `tf2onnx` stuff needs another version ... conda a separate container?

C:\Users\peter\Desktop\tface-gpu\tface.pip (tface-gpu)
λ conda create -n newtfonnx
λ conda activate newtfonnx
λ conda config --add channels conda-forge
λ conda config --add channels defaults
λ conda config --add channels nvidia
λ conda install cudatoolkit
λ conda install cudnn

... nowait; i need the old tensorflow to do this on windows ...

???

λ python -m venv tf2onnx_env
λ "tf2onnx_env/Scripts/activate"


pip install tf2onnx

... oh ... it doesn't line up ... maybe i do need WSL for linux

... or maybe i can keep doing training in the old version then conversion in a new one ...


tf2onnx_env\Scripts\activate


...

okay ... tflow is giving up on windows and using WSL2

fine?

i should learn that

https://www.youtube.com/watch?v=0S81koZpwPA


...

meanwhile, i got an onnx file out of it.

i have;
- conda to train the model on the GPU
- a venv to conver it to onnx

i need;
- load it into the baracuda stuff
- ? daapt input and output?


