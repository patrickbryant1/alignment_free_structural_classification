#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Parse and analyze CATH data .''')

parser.add_argument('sequences', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to clustered CATH sequences.')

parser.add_argument('outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####

def read_fasta(sequences):
    '''Read the fasta sequences and analyze the H-groups
    '''
    fetched_uids = []
    fetched_sequences = []
    with open(sequences, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line[0]=='>':
                #Create a new entry
                sequence = ''
                fetched_uids.append(line.split('|')[2].split('/')[0])
                pdb.set_trace()
            else:
                sequence+=line


#####MAIN#####
args = parser.parse_args()
sequences = args.sequences[0]
outdir = args.outdir[0]


read_fasta(sequences)
