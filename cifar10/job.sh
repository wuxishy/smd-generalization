#!/bin/bash

module load anaconda/2022a
source activate ffcv2
alias python=$HOME/.conda/envs/ffcv2/bin/python

cp *.beton $TMPDIR/

python train_cifar.py --config-file default_config.yaml \
                      --training.arch efficientnet \
                      --data.train_dataset $TMPDIR/cifar_train.beton \
                      --data.val_dataset $TMPDIR/cifar_test.beton \
                      2>&1 | tee output.txt
