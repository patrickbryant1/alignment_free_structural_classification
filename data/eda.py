#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
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
    sequence_lengths = []
    with open(sequences, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line[0]=='>':
                #Save seuqence
                if len(fetched_uids)>0:
                    fetched_sequences.append(sequence)
                    sequence_lengths.append(len(sequence))
                #Save uid
                fetched_uids.append(line.split('|')[2].split('/')[0])
                #Create a new entry
                sequence = ''

            else:
                sequence+=line



        #Append the last sequence
        fetched_sequences.append(sequence)
        sequence_lengths.append(len(sequence))


    df = pd.DataFrame()
    df['uid'] = fetched_uids
    df['sequence'] = fetched_sequences
    df['seqlen'] = sequence_lengths

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
            if line[0]=='#':
                continue
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

def analyze_distributions(df):
    '''analyze the C.A.T.H distributions
    '''
    #uid,sequence,Class,Architecture,Topology,H-group
    for param in ['Class','Architecture','Topology','H-group']:
        counted = Counter(df[param])
        groups = np.array([*counted.keys()])
        print(param,len(groups), 'groups')
        vals = np.array([*counted.values()])
        if param == 'H-group':
            print(len(vals[np.where(vals==1)]), 'out of', len(vals), 'H-groups with only one entry')
        sns.distplot(vals)
        plt.title(param)
        plt.savefig(outdir+param+'_hist.png', format='png', dpi=300)
        plt.close()

        sort_ind = np.argsort(vals)
        if len(groups) <10:
            plt.bar(groups[sort_ind],vals[sort_ind])
        else:
            plt.bar(np.arange(len(groups)),vals[sort_ind])
            plt.yscale('log')

        plt.xlabel('Group')
        plt.ylabel('Number of entries')
        plt.title(param)
        plt.savefig(outdir+param+'_bar.png', format='png', dpi=300)
        plt.close()


    #Plot seqlens
    sns.distplot(df['seqlen'])
    plt.title('Sequence length')
    plt.savefig(outdir+'seqlen_hist.png', format='png', dpi=300)
    plt.close()

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
#Analyze
analyze_distributions(df)
