# SRRAM
Unofficial Implementation of RAM: Residual Attention Module for Single Image Super-Resolution. 

Jun-Hyuk Kim, Jun-Ho Choi, Manri Cheon and Jong-Seok Lee,"RAM: Residual Attention Module for Single Image Super-Resolution", arXiv preprint arXiv:1811.12043, 2018 [[arXiv]](https://arxiv.org/abs/1811.12043)


## Requirements
- tensorflow 1.10+
- numpy
- scipy
- googledrivedownloader
- requests

## Downloading Data
In this repo, you can use General-100 dataset by the following commands. 
```
root $ cd ./data
data $ python General-100.py
```
Downloaded data will be randomly splitted into `train:test:val = 8:1:1`. If you want to use any other datasets or split strategies, please change the codes.

## Training
You can train the network like below.
```
python train.py
```
You can set the hyper-parameters therough the arguments. Please check them out by setting `--help`.  
The default parameters is not optimized now, so I will update them.

## Testing on your image
After your training, you can test the model on your image like below.
```
python test.py --model_dir ./saved/<PID> --filename path/to/your/image
```

## Results
**UNDER CONSTRUCTION**