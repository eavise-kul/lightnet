#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Transform annotations for VOCdevkit to the brambox pickle format
#

import os
import sys
import argparse
import xml.etree.ElementTree as ET
import brambox as bb

TRAINSET = [
    ('2012', 'train'),
    ('2012', 'val'),
    ('2007', 'train'),
    ('2007', 'val'),
    ]
TESTSET = [
    ('2007', 'test'),
    ]


def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = os.path.splitext(root.find('filename').text)[0]
    return f'{folder}/JPEGImages/{filename}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert annotations and split them in train/test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('root', help='Root data folder (Which contains VOCDevkit)')
    parser.add_argument('-v', '--verbose', help='Print debug messages', action='store_true')
    parser.add_argument('-d', '--difficult', help='Remove difficult training annotations', action='store_false')
    parser.add_argument('-x', '--extension', metavar='.EXT', help='Pandas extension to store data (See PandasParser)', default='.pkl')
    args = parser.parse_args()

    print('Getting training annotation filenames')
    train = []
    for (year, img_set) in TRAINSET:
        with open(f'{args.root}/VOCdevkit/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
            ids = f.read().strip().split()
        train += [f'{args.root}/VOCdevkit/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids]

    if args.verbose:
        print(f'\t{len(train)} xml files')

    print('Parsing training annotation files')
    train_annos = bb.io.load('anno_pascalvoc', train, identify)

    if args.difficult:
        if args.verbose:
            print(f'\tRemoving difficult annotations')
        train_annos = train_annos[~train_annos.difficult]

    print('Generating training annotation file')
    bb.io.save(train_annos, 'pandas', f'{args.root}/train{args.extension}')

    print()

    print('Getting testing annotation filenames')
    test = []
    for (year, img_set) in TESTSET:
        with open(f'{args.root}/VOCdevkit/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
            ids = f.read().strip().split()
        test += [f'{args.root}/VOCdevkit/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids]

    if args.verbose:
        print(f'\t{len(test)} xml files')

    print('Parsing testing annotation files')
    test_annos = bb.io.load('anno_pascalvoc', test, identify)

    if args.difficult:
        if args.verbose:
            print(f'\tRemoving difficult annotations')
        test_annos = test_annos[~test_annos.difficult]

    print('Generating testing annotation file')
    bb.io.save(test_annos, 'pandas', f'{args.root}/test{args.extension}')
