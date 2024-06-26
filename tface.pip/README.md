
https://chatgpt.com/c/5a811017-3ebd-4546-a84e-444a031bd5c0


setup like ...

```bash
> python -m venv pip_venv
> "pip_venv/Scripts/activate"
> python -m pip install --upgrade pip
> pip install -r requirements.txt
```

... then do something to copy the "wider face" ...

http://shuoyang1213.me/WIDERFACE/


maybe?

i'm going to "tran" the thing overnight and then ... i dhunno ... i'll see if I can convert it to a C#/Unity3D program


## check cuda/GPU is on

need cuda 8.0 and and cudnn 5.1 (i guess?)


```
import tensorflow as tf

# List all available devices
devices = tf.config.list_physical_devices()
print("Available devices:")
for device in devices:
    print(device)

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available())
```

....

trainy again with miniconda and cuda 12.5

.. added a check; the dlls aren't there after all!


# conda?

install conda
    ???

setup the outer environment;
    λ conda install -c conda-forge cudatoolkit=11 cudnn=8

activate the environment
    λ conda activate

install python 3.10 in the condan environment
    (base) λ conda install python=3.10

setup the pips packages
    (base) λ pip install -r requirements.txt

run the gpu-chekc
    (base) λ python gpucheck.py

