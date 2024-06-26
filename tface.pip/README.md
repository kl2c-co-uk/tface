
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

