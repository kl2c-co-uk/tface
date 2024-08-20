
"You Only Look Once Version5" (hereafter **yolo5**) is an image classification thingie written in PyTorch.
If you're coming here from kl2c (or anything using FaceONNX) it's what is used in FaceONNX (with some porcelain) to turn images into maps of facial patches.

So, since we want to detect human and cartoon faces; the attempt here is to
- (re) train the detector on a face dataset
- add/train cartoon face dataset as well
- add/train eyes (just to be safe) as a third class
- export it to `.onnx` with "opset_version=9" for [Unity's BarraCUDA tooling](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html)
    - BarraCUDA is hekking old, so, you need to tell the ONNX tools to speak in version 9 (IIRC they defaulted to 27 when I used them, but, happily spat out v9 when asked)
    - Untiy has a guide on this https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/Exporting.html
    - BarraCUDA seems to have a neat-trick to translate between GPU textures and Tensors which means; it's faster than FaceONNX!
- *maybe* build a second "supernetwork" to translate the yolo5 output to a heat map
    - i want MINIMAL trainable parameters
    - this should export to a `.onnx` file


## Setup / Training

You'll need [MiniConda](https://docs.anaconda.com/miniconda/miniconda-install/) and the [CUDA ToolKit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network) to use this, along with git.

> Peter L will only install or use via [Cmder](https://cmder.app/) there are probably more direct ways to do it, but, then involve more steps and configuration garbage.

> ... oh, you'll probably need an NVidia GPU (I think) but I've been able to run it with the K620s we have from iBit and a GTX 970 so "anything made after 2014" is probably fine if you're buying a new one ...
> > It's likely that non-NVidia stuff will transparently "just work" 

To use the Unity project, you need to install Unity 2021.3 (just open it with [Unity Hub]() and install the version) to avoid constantly tweaking the version numbers.

The following shell commands create a conda environment with the correct packages setup to run the training.


```
conda create --prefix target/pytorch python=3.8

conda activate target/pytorch

conda install -y pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia

pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install -r yolo5.txt
```


> the `target/` dir should be ignored by version control.  it gets really big. 

After this running `python yolo5.py` will run the trianing thing.

> once the environment is activated, you can do it all with `conda install -y pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia && pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt && pip install -r yolo5.txt && python yolo5.py`

> When running the command in the futre, make sure to use `conda activate target/pytorch` to switch to the environment you just created.

> train it for a long time and shutdown the PC `λ conda activate target/pytorch && python yolo5.py && shutdown /s /t 120`

If you (or future me) want/need to edit the file, `yolov5.py` you might enjoy using `nodemon --ignore target/ yolo5.py` to auto-re-run the script whenever you save it.

> I've been editing with [Visual Code](https://code.visualstudio.com/) please don't use NotePad - it's an exercise in self-harm at this point.



----
----
notes on what i'm doing
----


it seems to keep not quite working

that's not good

to fix this, i want to understand when it does work.

okay

- i've checked and "my own" things give silly results
- i want to try and extraqc the model from FaceONNX and see if that can be import/export and used



... actually ... no; visualising both models in nettron reveals we're doing very different things.

so.

so copying faceonnx was a mistake.

i need ... to ...

the output is 1575x6

x y w h class confidence?
























asked abotu this regardless

```markdown
https://stackoverflow.com/questions/78840000/yolo5-unity3d-barracuda

Is there documentation for what yolo-v5's output is?

We (or I) are trying to use YOLOv5 (for now) to detect faces in real time from an HDMI input.
Cartoon faces, video game faces, natural faces - all are welcome for our application.
I've followed the instructions to retrain it, but, the results seem nonsensical.

[![enter image description here][1]][1]

I've red the embedding/export instructions - but - don't understand the details of how to interpret the output tensor.

- https://github.com/Hexmagic/ONNX-yolov5/blob/master/src/test.cpp

TLDR; I'm trying to interpret the yolo-v5 output but the examples don't seem to illustrate what's going on to me.


  [1]: https://i.sstatic.net/wivN4y8Y.png
```



----



last round was interupted by windows update

i want a "retrain" option which points to a `.pt` file for this (and other) occaisions










https://blog.paperspace.com/train-yolov5-custom-data/
https://chatgpt.com/c/42665fa1-7076-4ad4-9e8e-c87a27eb5a53



pip install -r yolo5.txt



conda activate /path/to/your/env



dataset should be;

dataset/
    images/
        train/
            image1.jpg
            image2.jpg
            ...
        val/
            image1.jpg
            image2.jpg
            ...
    labels/
        train/
            image1.txt
            image2.txt
            ...
        val/
            image1.txt
            image2.txt
            ...


class_id x_center y_center width height


```
train: /path/to/your/dataset/images/train
val: /path/to/your/dataset/images/val

nc: 1  # number of classes
names:
  - face

```

