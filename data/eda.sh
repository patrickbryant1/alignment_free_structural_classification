#!/usr/bin/env bash
SEQUENCES=./cath-dataset-nonredundant-S20-v4_2_0.fa
DOMAINLIST=./cath-domain-list-S100-v4_2_0.txt
OUTDIR=../results/
./eda.py --sequences $SEQUENCES --domain_list $DOMAINLIST --outdir $OUTDIR
