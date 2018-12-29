# SRRAM
## Inttoduction
This is an unofficial implementation of RAM: Residual Attention Module for Single Image Super-Resolution [[arXiv]](https://arxiv.org/abs/1811.12043). 

RAM is one of attention blocks for super-resolution that has channel-attention(CA) and spatial-attention(SA). CA is computed using global variance pooling and fully-connected layers, and SA is obtained by depth-wise covolutions. After the parallel computation, they are fused by element-wise addition and activated by softmax operator, finally applied into the original feature maps. Before the output, there is a skip-connection from the previous layer.
<img width="733" alt="スクリーンショット 2018-12-16 15.15.10.png" src="https://qiita-image-store.s3.amazonaws.com/0/274187/3fdf24c9-5c5f-28d2-dfd4-3af4351c4c75.png">

SRRAM consists of stacked RAMs like EDSR.
<img width="733" alt="スクリーンショット 2018-12-16 15.15.10.png" src="https://camo.qiitausercontent.com/7d4bb73ca4d3522f38ecc9ae4f20c99e4c3664b5/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3237343138372f32616165653633632d396161632d333930362d323964662d6566393233326332656439322e706e67">

The attention blocks having CA and SA have already proposed, however, their attentions are applied sequentially or directly to input features, which means that the attentions is appied 2 times for each block. RAM is clearly different from other blocks because of the fusion before the application.
<img width="733" alt="スクリーンショット 2018-12-16 15.15.10.png" src="https://camo.qiitausercontent.com/adf2465d40117b830fd95fd1e90f73879cdf93a9/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3237343138372f32333731666162362d616263642d346132652d653936312d6132353632396538303336322e706e67">


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

## Citation
```
Jun-Hyuk Kim, Jun-Ho Choi, Manri Cheon and Jong-Seok Lee,"RAM: Residual Attention Module for Single Image Super-Resolution", arXiv preprint arXiv:1811.12043, 2018 
```