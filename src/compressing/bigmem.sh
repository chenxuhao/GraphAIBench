#!/bin/bash
srun -A xhchen_bigmem -N 1 -n 1 -c 32 --mem=1500G -p bigmem --time 1:00:00 --pty bash
