
> 2024-07-19; this is current one
> am trying to mak the unpacking facter


```bat
conda create --prefix target/pytorch python=3.8

conda activate target/pytorch

conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia

pip install -r yolo5.txt

python yolo5.py
```

... or if you're developing ...

```
nodemon --ignore target yolo5.py
```



## junk notes


https://blog.paperspace.com/train-yolov5-custom-data/
https://chatgpt.com/c/42665fa1-7076-4ad4-9e8e-c87a27eb5a53





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

