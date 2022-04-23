#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

mkdir -p output
python3 cifar_resnet.py 0.0  | tee output/outputSGD.txt
python3 cifar_resnet.py 0.9  | tee output/outputSGDM.txt
cp -r output azizan-lab_shared/generalisation
