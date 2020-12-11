# HW3-Instance-segmentation
## Environment
Pytorch 1.7.0  
Python 3.8.5  
Ubuntu 18.04  
## Install Detectron2
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
## Train
Download pretrained model weights at https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl  
Other models are avaliable at https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md  
Create a folder named 'models' and put the downloaded weights into it  
  
Extract the training data in this folder or set the DATA_ROOT variable in TrainMyRCNN.py  
Run training script  
```
python3 TrainMyRCNN.py
```

## Test
Run test scripts
```
python3 Test.py
```
The output will save to submission.json
