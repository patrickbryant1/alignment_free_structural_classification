#!/usr/bin/env bash
LRSLOSSES=../../results/lrs_losses.npy
OUTDIR=../../results/
./analyze.py --lrs_losses $LRSLOSSES --outdir $OUTDIR
