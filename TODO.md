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
  - [ ] Improve loss function on conceptual level

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

## Engine
Engine subpackage
  - [X] Create base engine class
  - [X] Test base engine

## Models
Model implementations subpackage
  - [X] Add yolo-voc
  - [ ] Add tiny-yolo
  - [X] Create darknet dataset
  - [X] Write documentation

## Varia
Various bits and bops
  - [X] Update README: credits to marvis
  - [ ] Update README: how to use
  - [X] Refactor organisation of the package
  - [ ] Test whether box+nms is faster than separate
  - [X] Add requirements.txt
  - [X] Write logger documentation

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
  - [ ] Compare accuracy yolo-voc trained weights

