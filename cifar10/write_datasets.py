from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', required=True),
    val_dataset=Param(str, 'Where to write the new dataset', required=True),
    test_dataset=Param(str, 'Where to write the new dataset', required=True),
)

@param('data.train_dataset')
@param('data.val_dataset')
@param('data.test_dataset')
def main(train_dataset, val_dataset, test_dataset):
    datasets = {
        'train': torchvision.datasets.CIFAR10('~/data', train=True, download=False),
        'test': torchvision.datasets.CIFAR10('~/data', train=False, download=False)
    }

    for (name, ds) in datasets.items():
        if name == 'train':
            # generate indices for train vs. validation split
            dataset_size = len(ds)
            train_size = int(dataset_size * 0.9)
            train_idx = np.random.choice(dataset_size, train_size, replace=False)
            val_idx = list(set(range(x.shape[0])) - set(s1))

            # construct train and validation datasets 
            train_path = train_dataset 
            train_writer = DatasetWriter(train_path, {
                'image': RGBImageField(), 
                'label': IntField()
            })
            train_writer.from_indexed_dataset(ds, train_idx)

            val_path = val_dataset 
            val_writer = DatasetWriter(val_path, {
                'image': RGBImageField(), 
                'label': IntField()
            })
            val_writer.from_indexed_dataset(ds, val_idx)
        else: 
            path = test_dataset
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
