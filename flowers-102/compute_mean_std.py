#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:30:14 2022

@author: haimoshri
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

tranform = transforms.Compose([transforms.Resize((256, 256)),
                               transforms.ToTensor()])

dataset = datasets.Flowers102(root = "../../azizan-lab_shared/datasets", split = 'train', transform = tranform)


loader = DataLoader(dataset,
                         batch_size=128,
                         num_workers=0,
                         shuffle=False)

mean = 0.
std = 0.
for images, _ in loader:
    print("Here")
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)

print(mean*255)
print(std*255)
