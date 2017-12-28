#!/usr/bin/env python
#
#   Visualise annotations
#   Copyright EAVISE
#

import os
import brambox.boxes as bbb

ROOT = '.data'
SET = 'train.pkl'


if __name__ == '__main__':
    annos = bbb.parse('anno_pickle', f'{ROOT}/{SET}')
    bbb.show_bounding_boxes(annos, f'{ROOT}/VOCdevkit', '', True)
