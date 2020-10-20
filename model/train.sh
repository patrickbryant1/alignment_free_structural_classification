#!/usr/bin/env bash
SEQDF=../data/seqdf.csv
OUTDIR=../results/

SINGULARITY=/opt/singularity3/bin/singularity
SINGIMAGE=/home/pbryant/singularity_ims/tf13.sif

$SINGULARITY run --nv $SINGIMAGE python ./model.py --sequence_df $SEQDF --outdir $OUTDIR
