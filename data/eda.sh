#!/usr/bin/env bash
SEQUENCES=./cath-domain-seqs-S95.fa
DOMAINLIST=./cath-domain-list.txt
OUTDIR=../results/
./eda.py --sequences $SEQUENCES --domain_list $DOMAINLIST --outdir $OUTDIR
