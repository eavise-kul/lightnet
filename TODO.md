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
  - [X] Explicit BGR2RGB transform for cv2

## Engine
Engine subpackage
  - [X] Create base engine class
  - [X] Test base engine
  - [X] Engine rates
  - [X] Rework visualisation concept
  - [X] Make visdom completely optional _(fix imports)_
  - [X] Rework entire engine class
  - [X] Rework logging mechanism to use standard python logger

## Models
Model implementations subpackage
  - [X] Add yolo-voc
  - [X] Add tiny-yolo
  - [X] Create darknet dataset
  - [X] Write documentation
  - [X] Add Darknet19
  - [X] Add Mobilenet YOLO

## Varia
Various bits and bops
  - [X] Update README: credits to marvis
  - [X] Refactor organisation of the package
  - [X] Add requirements.txt
  - [X] Write logger documentation
  - [X] Add logger file print
  - [X] Add brambox intersphinx mapping
  - [ ] Add _how to_ guide
  - [ ] Add _examples_ guide
  - [ ] Rewrite _score_ guide with new numbers
  - [ ] Rewrite _speed_ guide with new numbers

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
  - [X] Compare accuracy yolo-voc trained weights
  - [ ] Compare speed yolo-voc training
  - [ ] Rework examples to work with new lightnet API

