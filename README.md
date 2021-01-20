<img src="https://gitlab.com/EAVISE/lightnet/raw/master/docs/.static/lightnet-long.png" alt="Logo" width="100%">  

Building blocks to recreate Darknet networks in Pytorch  
[![Version][version-badge]][release-url]
[![Documentation][doc-badge]][documentation-url]
[![PyTorch][pytorch-badge]][pytorch-url]
[![Pipeline][pipeline-badge]][pipeline-url]
<a href="https://ko-fi.com/D1D31LPHE"><img alt="Ko-Fi" src="https://www.ko-fi.com/img/githubbutton_sm.svg" height="20"></a>  
[![VOC][voc-badge]][voc-url]
[![COCO][coco-badge]][coco-url]



## Why another framework
[pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) is working perfectly fine,
but does not easily allow a user to modify an existing network.  
This is why I decided to create a library,
that gives the user all the necessary building blocks, to recreate any darknet network.
This library has everything you need to control your network,
weight loading & saving, datasets, dataloaders and data augmentation.

Where it started as library to recreate the darknet networks in PyTorch,
it has since grown into a more general purpose single-shot object detection library.

## Installing
First install [PyTorch and Torchvision](https://pytorch.org/get-started/locally).  
Then clone this repository and run one of the following commands:
```bash
# If you just want to use Lightnet
pip install brambox   # Optional (needed for training)
pip install .

# If you want to develop Lightnet
pip install -r develop.txt
```
> This project is python 3.6 and higher so on some systems you might want to use 'pip3.6' instead of 'pip'

## How to use
[Click Here](https://eavise.gitlab.io/lightnet) for the API documentation and guides on how to use this library.  
The _examples_ folder contains code snippets to train and test networks with lightnet. For examples on how to implement your own networks, you can take a look at the files in _lightnet/models_.
>If you are using a different version than the latest,
>you can generate the documentation yourself by running `make clean html` in the _docs_ folder.
>This does require some dependencies, like Sphinx.
>The easiest way to install them is by using the __-r develop.txt__ option when installing lightnet.

## Cite
If you use Lightnet in your research, please cite it.
```
@misc{lightnet18,
  author = {Tanguy Ophoff},
  title = {Lightnet: Building Blocks to Recreate Darknet Networks in Pytorch},
  howpublished = {\url{https://gitlab.com/EAVISE/lightnet}},
  year = {2018}
}
```

## Main Contributors
Here is a list of people that made noteworthy contributions and helped to get this project where it stands today!

- [Tanguy Ophoff](https://gitlab.com/0phoff)
- [Jon Crall](https://gitlab.com/Erotemic)
- [Cedric Gullentops](https://github.com/CedricGullentops)


[version-badge]: https://img.shields.io/pypi/v/lightnet.svg?label=version
[doc-badge]: https://img.shields.io/badge/-Documentation-9B59B6.svg
[pytorch-badge]: https://img.shields.io/badge/PyTorch-1.5.0-F05732.svg
[pipeline-badge]: https://gitlab.com/EAVISE/lightnet/badges/master/pipeline.svg
[release-url]: https://gitlab.com/EAVISE/lightnet/tags
[documentation-url]: https://eavise.gitlab.io/lightnet
[pytorch-url]: https://pytorch.org
[pipeline-url]: https://pypi.org/project/lightnet
[voc-badge]: https://img.shields.io/badge/repository-Pascal%20VOC-%2300BFD8
[voc-url]: https://gitlab.com/eavise/top/voc
[coco-badge]: https://img.shields.io/badge/repository-MS%20COCO-%2300BFD8
[coco-url]: https://gitlab.com/eavise/top/coco
