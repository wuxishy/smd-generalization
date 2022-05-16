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
    test_dataset=Param(str, 'Where to write the new dataset', required=True)
)

@param('data.raw_data')
@param('data.train_dataset')
@param('data.test_dataset')
def main(raw_data, train_dataset, test_dataset):
    raw_data = os.path.expandvars(raw_data)
    datasets = {
        'train': torchvision.datasets.CIFAR10(raw_data, train=True, download=False),
        'test': torchvision.datasets.CIFAR10(raw_data, train=False, download=False)
    }

    for (name, ds) in datasets.items():
        if name == 'train':
            # construct train and validation datasets 
            path = os.path.expandvars(train_dataset)
        else: 
            path = os.path.expandvars(test_dataset)
        
        writer = DatasetWriter(path, {
            'image': RGBImageField(), 
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()
