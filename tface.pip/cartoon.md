

retring the no-cartoon settings with cartoons


> python yolo5.py --weights "C:/Users/peter/Desktop/tface-train-nocartoon/tface.pip/target/yolov5/runs/train/exp5/weights/best.pt" && shutdown /s /t 180

.. this finished around 9pm and ... works ... mostly ... it finds the human and sometimes the blue cluies animals.

`yoloface.2024-08-21.192-9pm.onnx`

So now I want to train for another 40 iterations picking up where this left off.
Might be an idea to audit more cartoons.

> python yolo5.py --weights "C:/Users/peter/Desktop/tface-train-nocartoon/tface.pip/target/yolov5/runs/train/exp6/weights/best.pt" && shutdown /s /t 180

memory usage is ...
```
192x48 = 1.3G (in the first ones)
192x64 = 1.7G on epoc 1/39 so i assume it'll stay under 3G
```

----











https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=cartoon+face+detection&oq=cartoon+face



https://dl.acm.org/doi/abs/10.1145/3394171.3413726

https://arxiv.org/pdf/1907.13394

@inproceedings{zheng2020cartoon,
  title={Cartoon face recognition: A benchmark dataset},
  author={Zheng, Yi and Zhao, Yifan and Ren, Mengyuan and Yan, He and Lu, Xiangju and Liu, Junhui and Li, Jia},
  booktitle={Proceedings of the 28th ACM international conference on multimedia},
  pages={2264--2272},
  year={2020}
}


https://iqiyi.cn/icartoonface

https://github.com/luxiangju-PersonAI/iCartoonFace

https://drive.google.com/drive/folders/1ARKrhmGAMwVNr8M9kXgDzMUDhzusLxb7

https://drive.google.com/drive/folders/1ARKrhmGAMwVNr8M9kXgDzMUDhzusLxb7

looks like 90k which might be good enough





iCartoon






from datasource import Blurb, Cache, md5, ZipWalk, ensure_directory_exists

cache = Cache('target/')

annotations = cache.download(
    # annotations = target/712e3f96290bfc9c1c93a18f16ef40e8
    'https://drive.usercontent.google.com/download?id=15IHSlNBZBZs_hj6B341swc00ha5fpvB7&export=download&authuser=0&confirm=t&uuid=72fd55fe-6a76-4c73-91ee-63de54aa2775&at=APZUnTXR3ogM4tIFCGrcFoaswAor:1723670943322'
)
print('annotations = '+annotations)

valuation = cache.download(
    # valuation = target/acf340e2b032e96f331897b2f15ba3b9
    'https://drive.usercontent.google.com/download?id=111cgWh3Z1QBviMMahAGwPKpR3IlNCrsd&export=download&authuser=0&confirm=t&uuid=6857a0ed-1f1e-4fa0-ba65-d78db1771692&at=APZUnTVN1sqA71HIDRV3c5o7oHyy:1723671362710'
)
print('valuation = '+valuation)


training = cache.download(
    # training = target/cb67961c4ba344c84b3e5442206436ac
    'https://drive.usercontent.google.com/download?id=1xXpE0qs2lONWKL5dqaFxqlJ_t5-glNpg&export=download&authuser=0&confirm=t&uuid=f6f6beb7-4c3b-40a7-b52d-12c62c2e84fe&at=APZUnTV9QwxtWfOsgjgqW-7icoaM:1723671279280'
)
print('training = '+training)




i don't know what teh value set is




an unresolved issue; at least one of the WIDER images (the third one in the training set) has a cartoon in the background



i'm going to retrain it from the previous BEST weights (even with the new resolution) and see ... see what's what.



