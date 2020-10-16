#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import pandas as pd
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Parse and analyze CATH data .''')

parser.add_argument('--sequences', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to clustered CATH sequences.')
parser.add_argument('--domain_list', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to C.A.T.H relation for CATH identifiers.')
parser.add_argument('--outdir', nargs=1, type= str,
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
                #Save seuqence
                if len(fetched_uids)>0:
                    fetched_sequences.append(sequence)
                #Save uid
                fetched_uids.append(line.split('|')[2].split('/')[0])
                #Create a new entry
                sequence = ''

            else:
                sequence+=line
                print(sequence)


        #Append the last sequence
        fetched_sequences.append(sequence)


    df = pd.DataFrame()
    df['uid'] = fetched_uids
    df['sequence'] =fetched_sequences

    return df

def read_domain_list(domain_list):
    '''Read the screwed up tsv format from the domain list
    '''
    #Save
    fetched_uids = []
    fetched_classes = []
    fetched_architectures = []
    fetched_topologies = []
    fetched_hgroups = []
    #Loop through file
    with open(domain_list, 'r') as file:
        for line in file:
            line = line.split()
            fetched_uids.append(line[0])
            fetched_classes.append(line[1])
            fetched_architectures.append('.'.join(line[1:3]))
            fetched_topologies.append('.'.join(line[1:4]))
            fetched_hgroups.append('.'.join(line[1:5]))

    df = pd.DataFrame()
    df['uid'] = fetched_uids
    df['Class'] = fetched_classes
    df['Architecture'] = fetched_architectures
    df['Topology'] = fetched_topologies
    df['H-group'] = fetched_hgroups

    return df
#####MAIN#####
args = parser.parse_args()
sequences = args.sequences[0]
domain_list = args.domain_list[0]
outdir = args.outdir[0]

#Read domain list
hgroup_df = read_domain_list(domain_list)

sequence_df = read_fasta(sequences)
#Join on uid
df = pd.merge(sequence_df,hgroup_df,on='uid', how='left')
#Save
df.to_csv('seqdf.csv')
