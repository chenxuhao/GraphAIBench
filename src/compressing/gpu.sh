#!/bin/bash
srun -A xhchen_gpu -N 1 -n 1 -c 1 --gpus=1 --mem=250G -p ica100 --time 01:00:00 --pty bash
