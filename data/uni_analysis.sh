#!/usr/bin/env bash
DATA=/hdd/pbryant/data/alignment_free_structural_classification/all/uniprot_proteome_reference_gene3d.tab
OUTDIR=/hdd/pbryant/data/alignment_free_structural_classification/all/
./uniprot_analysis.py --uniprot_fetch $DATA --outdir $OUTDIR
