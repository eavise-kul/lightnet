Training YOLO on VOC
====================
This folder contains the scripts needed to train the lightnet YOLO network on VOC.  
We perform the same training and testing as explained on the [yolo website](https://pjreddie.com/darknet/yolo/#train-voc).


## Get the data
We train YOLO on all of the VOC data from 2007 and 2012.
To get all the data, make a directory to store it all, and execute the following commands:
```bash
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
There will now be a VOCdevkit folder with all the data.


## Generating labels
We need to have the right labels for training and testing the network.  
While brambox (and thus lightnet) can work with Pascal VOC annotations,
we still need to group the data in a training and testing set.
Because we are converting this anyway, we take the opportunity to convert the annotations to a pickle format,
which will be faster to parse whilst training/testing.
```bash
# Change the 'ROOT' variable in labels.py to point to the root directory that contains VOCdevkit
./labels.py
```
You can check whether to annotation conversion was succesfull, by running the __visual_anno.py__ script.

> Note that there is no validation set.
> We perform the same training cycle as darknet, and thus have no testing whilst training the network.
> This means there is no need for a separate testing and validation set,
> but also means we have no way to check how well the network performs whilst it is training.


## Get weights
For training, we use weights that are pretrained on imagenet.
```bash
wget https://pjreddie.com/media/files/darknet19_448.conv.23
```


## Train model
Use the __train.py__ script to train the model. You can use _train.py --help_ for an explanation of the arguments and flags.
```bash
# Adapt the model parameters inside of train.py to suite your needs
./train.py -cv darknet19_448.conv.23 data/train.pkl
```
> You cannot train with multiple workers (yet).
> There is a small problem with how workers are implemented and the need for randomly resizing the input.
> I hope to fix this issue soon, you can track this [issue](https://github.com/pytorch/pytorch/issues/4382) on the pytorch github repo.


## Test model
Use the __test.py__ script to test the model. You can again use _test.py --help_ for an explanation of the arguments and flags.
```bash
# We use tqdm for a nice loading bar
pip install tqdm 

# Adapt the model parameters inside of test.py to suite your needs
./test.py -cv backup/weight_40000.pt data/test.pkl
```
