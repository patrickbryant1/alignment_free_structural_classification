#!/usr/bin/env bash
SEQDF=../data/seqdf.csv
OUTDIR=../results/

./model.py --sequence_df $SEQDF --outdir $OUTDIR
