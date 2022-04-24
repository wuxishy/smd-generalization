# smd-generalization
To install FFCV:

First do `conda init bash` if this is your first time setting up a conda enviornment, then:

`conda create -n ffcv python=3.8 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge && conda activate ffcv && pip install ffcv`

