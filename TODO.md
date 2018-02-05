# Lightnet
Lightnet TODO list  
Created and maintained with [TodoCMD](https://github.com/0phoff/TodoCMD). _Shameless self-promotion_

## Network
Network subpackage
  - [X] Add darknet layers
  - [X] Add/Test darknet weight loading/saving
  - [X] Add/Test pickle weight loading/saving
  - [X] Add/Test loss function
  - [X] Improve loss function with py36 and pytorch 0.3.0
  - [X] Improve loss function on conceptual level
  - [X] Loss function can work with brambox objects

## Data
Data subpackage
  - [X] Add generic BramboxDataset
  - [X] Add/Test letterbox transform
  - [X] Add/Test crop transform
  - [X] Add/Test flip transform
  - [X] Add/Test HSV transform
  - [X] Add/Test postprocessing
  - [X] Improve postprocessing (box conversion + nms in one loop)
  - [X] Write documentation
  - [X] Dataset always works with brambox objects
  - [X] Improve non-maximum suppression _(gpu/pytorch)_
  - [X] Improve bbox_to_brambox _(gpu/pytorch)_
  - [X] FIX preprocessing and random resizing with multiple workers
  - [X] Clean up dataset and preprocessing
  - [ ] RandomCrop: black border -> gray 

## Engine
Engine subpackage
  - [ ] Rework engine (variables)
  - [X] Create base engine class
  - [X] Test base engine
  - [X] Engine rates
  - [ ] Rework visualisation concept
  - [ ] Expose _log.open_file_
  - [ ] Make visdom completely optional _(fix imports)_

## Models
Model implementations subpackage
  - [X] Add yolo-voc
  - [X] Add tiny-yolo
  - [X] Create darknet dataset
  - [X] Write documentation

## Varia
Various bits and bops
  - [X] Update README: credits to marvis
  - [X] Refactor organisation of the package
  - [X] Add requirements.txt
  - [X] Write logger documentation
  - [X] Add logger file print
  - [ ] Add brambox intersphinx mapping
  - [ ] Add _how_to_use_ guide

## Examples and Scripts
Everything about creating examples and scripts to show off the library
  - [X] Add single image test
  - [X] Improve single image with transforms
  - [X] Test yolo-voc with 1 class
  - [X] Compare speed yolo-voc with 1 class
  - [X] Compare speed default yolo-voc
  - [X] Test default yolo-voc model
  - [X] Add training with yolo-voc 
  - [X] Add yolo-voc training with darknet engine
  - [X] Test yolo-voc training
  - [X] Test yolo-voc engine training
  - [ ] Compare speed yolo-voc training
  - [X] Compare accuracy yolo-voc trained weights

