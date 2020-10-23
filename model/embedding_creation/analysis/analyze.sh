#!/usr/bin/env bash
LRSLOSSES=../../../results/lrs_losses.npy
OUTDIR=../../../results/test_net/
#./lr_analysis.py --lrs_losses $LRSLOSSES --outdir $OUTDIR


#Look at the performance in one-shot learning.
JSON=../../../results/test_net/model.json
WEIGHTS=../../../results/test_net/weights-400-.hdf5
SEQDF=../../../data/seqdf_s100.csv
OUTDIR=../../../results/test_net/
./oneshot.py --json_file $JSON --weights $WEIGHTS --sequence_df $SEQDF --outdir $OUTDIR
