#!/bin/bash

module load anaconda/2022a
source activate ffcv
alias python=$HOME/.conda/envs/ffcv/bin/python

cp *.beton $TMPDIR/

python train_flowers.py --config-file default_config.yaml \
                      --training.arch efficientnet \
                      --data.train_dataset $TMPDIR/flowers-102_train.beton \
                      --data.val_dataset $TMPDIR/flowers-102_test.beton \
                      2>&1 | tee output.txt
