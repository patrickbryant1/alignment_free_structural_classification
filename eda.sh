#!/usr/bin/env bash
SEQUENCES=./cath-dataset-nonredundant-S20.fa
DOMAINLIST=./cath-domain-list-S100-v4_2_0.txt
./eda.py --sequences $SEQUENCES --domain_list $DOMAINLIST --outdir ./
