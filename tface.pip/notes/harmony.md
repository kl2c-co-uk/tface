

# notes for training on Peter's host "harmony"


```
peter@harmony:~/tface/tface.pip$ source target/pytorch/bin/activate

???


```


## cuda setup




https://gist.github.com/primus852/b6bac167509e6f352efb8a462dcf1854#file-cuda_11-7_installation_on_ubuntu_22-04-L29
sudo apt update


uggghhh

follow nvidia's

https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local

lloks like ... 3.11 has a bug ... so try 3.9? https://github.com/ultralytics/ultralytics/issues/8509

```
python3 -m venv target/pytorch

... no ...

```


```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

then

chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init

???

once cuda is availble?

