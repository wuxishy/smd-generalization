#!/bin/bash

#SBATCH --gres=gpu:volta:1

#Run the script
module load anaconda

mkdir -p output
python3 cifar_resnet.py 0.0  | tee output/outputSGD.txt
python3 cifar_resnet.py 0.9  | tee output/outputSGDM.txt
cp -r output azizan-lab_shared/generalisation
