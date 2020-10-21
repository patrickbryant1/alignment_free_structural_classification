#!/usr/bin/env bash
SEQDF=../data/seqdf.csv
PARAMS=./params/32_10_1_3_10_10.params
OUTDIR=../results/

SINGULARITY=/opt/singularity3/bin/singularity
SINGIMAGE=/home/pbryant/singularity_ims/tf13.sif

$SINGULARITY run --nv $SINGIMAGE python ./model.py --sequence_df $SEQDF --params_file $PARAMS --outdir $OUTDIR
