#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021a

# Run twice to check reproducibility
mkdir -p output
python3 cifar_resnet.py 0.0  | tee output/outputSMD_run1.txt
python3 cifar_resnet.py 0.0  | tee output/outputSMD_run2.txt
# python3 cifar_resnet.py 0.0  | tee output/outputSGD_run2.txt
# python3 cifar_resnet.py 0.9  | tee output/outputSGDM_run2.txt
# cp -r output azizan-lab_shared/generalisation
