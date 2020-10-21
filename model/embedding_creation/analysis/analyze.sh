#!/usr/bin/env bash
LRSLOSSES=../../../results/lrs_losses.npy
OUTDIR=../../../results/
./lr_analysis.py --lrs_losses $LRSLOSSES --outdir $OUTDIR
