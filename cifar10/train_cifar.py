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
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

Section('training', 'Hyperparameters').params(
    arch=Param(str, 'CNN architecture to use', required=True),
    pnorm=Param(float, 'p-value to use in SMD', required=True),
    # lr_init=Param(float, 'The initial learning rate to use', required=True),
    lr=Param(float, 'The maximum learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    # lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=128),
    num_workers=Param(int, 'The number of workers', default=8),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    test_dataset=Param(str, '.dat file to use for evaluation', required=True),
    output_directory=Param(str, 'directory to save outputs', required=True)
)

Section('checkpoint', 'for when training from a checkpoint').params(
    from_checkpoint=Param(bool, 'whether to train from a checkpoint', required=True),
    trial=Param(int, 'which trial number to grab checkpoint from', required=True)
)

@param('data.train_dataset')
@param('data.test_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, test_dataset=None, 
            batch_size=None, num_workers=None):
    paths = {
        'train': os.path.expandvars(train_dataset),
        'test': os.path.expandvars(test_dataset)
    }   

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.950, 113.865]
    CIFAR_STD = [62.993, 62.089, 66.705]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        # loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
        #                        order=ordering, drop_last=(name == 'train'),
        #                        pipelines={'image': image_pipeline, 'label': label_pipeline})
        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=False,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time

@param('training.arch')
def construct_model(output_directory, arch=None):
    if arch == 'resnet':
        model = ResNet18()
    elif arch == 'vgg':
        model = VGG('VGG11')
    elif arch == 'mobilenet':
        model = MobileNetV2()
    elif arch == 'efficientnet':
        model = EfficientNetB0()
    elif arch == 'regnet':
        model = RegNetX_200MF()
    model = model.to(memory_format=ch.channels_last).cuda()

    return model

@param('training.lr')
@param('training.pnorm')
@param('training.epochs')
@param('checkpoint.from_checkpoint')
@param('checkpoint.trial')
def train(model, loaders, output_directory, log_file = sys.stdout, lr=None, 
        epochs=None, pnorm=None, from_checkpoint=False, trial=0):
    #opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    #opt = Adam(model.parameters(), lr=lr)
    print("learning rate =", lr)
    opt = SMD_opt.SMD_qnorm(model.parameters(), lr=lr, q=pnorm)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    best_test_acc = 0

    start_epoch = 0
    if from_checkpoint:
        checkpoint = torch.load(f'{output_directory}/checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        print(f"Training from epoch {start_epoch+1}; current best test acc = {best_test_acc}", file=log_file)

    for epoch in tqdm(range(start_epoch, epochs)):
        total_correct = 0
        total_num = 0

        model.train()

        for i, (ims, labs) in enumerate(loaders['train']):
            # Train with a fixed learning rate
            # for param_group in opt.param_groups:
            #     param_group['lr'] = lr_schedule[epoch * iters_per_epoch + i]

            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

                total_correct += out.argmax(1).eq(labs).sum().cpu().item() 
                total_num += ims.shape[0]

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        #scheduler.step()

        if (epoch + 1) % 10 == 0:  
            print(total_correct, total_num)

            print(f'Epoch {epoch+1} loss: {loss.item():.4f}', file = log_file)
            train_acc = evaluate(model, loaders, 'train')
            print(f'Epoch {epoch+1} train acc: {train_acc:.4f}', file = log_file)
            test_acc = evaluate(model, loaders, 'test')
            print(f'Epoch {epoch+1} test acc: {test_acc:.4f}', file = log_file)
            log_file.flush()

            if test_acc > best_test_acc:
                print("Saving best model...", file=log_file)
                best_test_acc = test_acc
                torch.save(model, f'{output_directory}/best_model.pt')
        
        if (epoch + 1) == 1:
            print("Checkpointing...", file=log_file)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_test_acc': best_test_acc,
            }, f'{output_directory}/checkpoint.pt')
            
        if epoch + 1 >= 200:
            # train_acc = evaluate(model, loaders, 'train')
            if total_correct / total_num < 0.12:
                print("Training is too slow. Quitting.", file=log_file)
                break
        
        if total_correct == total_num:
            print(f"Reached 0 training error. Ending training on epoch {epoch+1}.", file=log_file)
            break

def evaluate(model, loaders, name):
    model.eval()
    # see if this is the best model so far         
    total_correct, total_num = 0, 0
    with ch.no_grad():
        for ims, labs in loaders[name]:
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item() 
                total_num += ims.shape[0]

    # print(f"{total_correct} out of {total_num} correct", file=log_file)
    return total_correct / total_num

def validation(model, loaders):
    accuracies = {}
    
    for name in ['train', 'test']:
        accuracies[name] = evaluate(model, loaders, name)
        print(f'{name} accuracy: {accuracies[name] * 100:.2f}%')
    
    return accuracies

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("cuda enabled!")

    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    if config['checkpoint.from_checkpoint']:
        print("checkpointing!")
        os.environ["OUTPUT"] = "$GROUP/smd-experiment/cifar10_fixed-lr-long-run/" + str(config['checkpoint.trial'])
    
    output_directory = os.path.expandvars(config['data.output_directory'])
    print(output_directory)
    output_directory = os.path.expandvars(output_directory)
    print(output_directory)

    if not config['checkpoint.from_checkpoint']:
        print("new training!")
        os.makedirs(output_directory, exist_ok = True)

    loaders, start_time = make_dataloaders()
    model = construct_model(output_directory)
    # save initial model so we can view weights later
    if not config['checkpoint.from_checkpoint']:
        torch.save(model, f'{output_directory}/init_model.pt')

    with open(f'{output_directory}/log.txt', 'a') as log_file:
        log_file.write('\n')
        train(model, loaders, output_directory, log_file)

    accuracies = validation(model, loaders)
    acc_file = f'{output_directory}/accuracy.yaml'
    with open(acc_file, 'w') as file:
        yaml.dump(accuracies, file)
    
    torch.save(model, f'{output_directory}/final_model.pt')
    
    print(f'Total time (min): {(time.time() - start_time) / 60:.2f}')
