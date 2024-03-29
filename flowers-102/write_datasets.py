from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import os, os.path

import torch as ch
import torch.utils.data as data
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

Section('data', 'arguments to give the writer').params(
    raw_data=Param(str, 'Where the raw data can be found', required=True),
    train_dataset=Param(str, 'Where to write the new dataset', required=True),
    val_dataset=Param(str, 'Where to write the new dataset', required=True),
    test_dataset=Param(str, 'Where to write the new dataset', required=True)
)

@param('data.raw_data')
@param('data.train_dataset')
@param('data.val_dataset')
@param('data.test_dataset')
def main(raw_data, train_dataset, val_dataset, test_dataset):
    raw_data = os.path.expandvars(raw_data)
    datasets = {
        'train': torchvision.datasets.Flowers102(raw_data, split='train'),
        'val': torchvision.datasets.Flowers102(raw_data, split='val'),
        'test': torchvision.datasets.Flowers102(raw_data, split='test')
    }
    data_path = {
        'train': os.path.expandvars(train_dataset),
        'val': os.path.expandvars(val_dataset),
        'test': os.path.expandvars(test_dataset),
    }

    for (name, ds) in datasets.items():
        train_path = os.path.expandvars(data_path[name])
        train_writer = DatasetWriter(train_path, {
            'image': RGBImageField(), 
            'label': IntField()
        })
        train_writer.from_indexed_dataset(ds)

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast Flowers-102 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()
