"""
Modified from FFCV example at https://docs.ffcv.io/ffcv_examples/cifar10.html.

First, from the same directory, run:

    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`

to generate the FFCV-formatted versions of CIFAR.

Then, simply run this to train models with default hyperparameters:

    `python train_cifar.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_cifar.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""
from argparse import ArgumentParser
from typing import List
import time
import numpy as np
import yaml
from tqdm import tqdm
import sys, os, os.path
from copy import deepcopy

from models import *

import SMD_opt

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, Adam, lr_scheduler
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, Resize, RandomCrop
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

Section('training', 'Hyperparameters').params(
    arch=Param(str, 'CNN architecture to use', required=True),
    pnorm=Param(float, 'p-value to use in SMD', required=True),
    lr_init=Param(float, 'The initial learning rate to use', required=True),
    lr=Param(float, 'The maximum learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=128),
    num_workers=Param(int, 'The number of workers', default=8),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    test_dataset=Param(str, '.dat file to use for evaluation', required=True),
    output_directory=Param(str, 'directory to save outputs', required=True)
)

@param('data.train_dataset')
@param('data.val_dataset')
@param('data.test_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None, test_dataset=None, 
            batch_size=None, num_workers=None):
    paths = {
        'train': os.path.expandvars(train_dataset),
        'val': os.path.expandvars(val_dataset),
        'test': os.path.expandvars(test_dataset)
    }   

    start_time = time.time()
    FLOWERS_MEAN = [110.4046, 97.3915, 75.5849]
    FLOWERS_STD = [66.1458, 53.5585, 56.5548]
    loaders = {}

    for name in ['train', 'val', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                Resize((256, 256)),
                RandomCrop((224, 224)),
                RandomTranslate(padding=2, fill=tuple(map(int, FLOWERS_MEAN))),
                Cutout(4, tuple(map(int, FLOWERS_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(FLOWERS_MEAN, FLOWERS_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time

@param('training.arch')
def construct_model(arch):
    if arch == 'resnet':
        model = ResNet18()
    elif arch == 'mobilenet':
        model = MobileNetV2()
    elif arch == 'efficientnet':
        model = EfficientNetB0()
    elif arch == 'regnet':
        model = RegNetX_200MF()
    model = model.to(memory_format=ch.channels_last).cuda()
    return model

@param('training.lr_init')
@param('training.lr')
@param('training.pnorm')
@param('training.epochs')
@param('training.lr_peak_epoch')
def train(model, loaders, log_file = sys.stdout, lr_init=None, lr=None, 
        epochs=None, lr_peak_epoch=None, pnorm=None):
    #opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #opt = Adam(model.parameters(), lr=lr)
    opt = SMD_opt.SMD_qnorm(model.parameters(), lr=lr, q=pnorm)
    #iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    '''
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    '''
    lr_schedule = np.interp(np.arange(epochs+1),
                            [1, lr_peak_epoch//5+1, lr_peak_epoch, epochs],
                            [lr_init, lr_init, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    #scheduler = lr_scheduler.StepLR(opt, 20, 0.5)
    #scheduler = lr_scheduler.CyclicLR(opt, lr*1e-3, lr, cycle_momentum=False) 
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    best_acc = 0.0
    best_model_state = None

    for epoch in tqdm(range(epochs)):
        for i, (ims, labs) in enumerate(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        scheduler.step()

        if (epoch + 1) % 10 == 0:  
            val_acc = validation(model, loaders)
            print(f'Epoch {epoch+1} validation: {val_acc * 100:.2f}%', file = log_file)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                print(f'Epoch {epoch+1}: best so far', file = log_file)
            
            model.train()   # return to training
    
    # return best model 
    return best_model_state

def evaluate(model, loaders):
    model.eval()
    accuracies = {}
    with ch.no_grad():
        for name in ['train', 'val', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in loaders[name]:
                with autocast():
                    out = model(ims)
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            accuracies[name] = total_correct / total_num
            print(f'{name} accuracy: {total_correct / total_num * 100:.2f}%')
    
    return accuracies

def validation(model, loaders):
    model.eval()
    # see if this is the best model so far         
    total_correct, total_num = 0, 0
    with ch.no_grad():
        for ims, labs in loaders['val']:
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item() 
                total_num += ims.shape[0]

    return total_correct / total_num

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast Flowers-102 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    output_directory = os.path.expandvars(config['data.output_directory'])
    os.makedirs(output_directory, exist_ok = True)

    loaders, start_time = make_dataloaders()
    model = construct_model()
    with open(f'{output_directory}/log.txt', 'w') as log_file:
        best_model_state = train(model, loaders, log_file)

    model.load_state_dict(best_model_state)

    accuracies = evaluate(model, loaders)
    acc_file = f'{output_directory}/accuracy.yaml'
    with open(acc_file, 'w') as file:
        yaml.dump(accuracies, file)
    
    torch.save(best_model_state, f'{output_directory}/model.pt')
    
    print(f'Total time (min): {(time.time() - start_time) / 60:.2f}')
