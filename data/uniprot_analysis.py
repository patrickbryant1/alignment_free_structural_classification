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

parser.add_argument('--uniprot_fetch', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to uniprot accessions with CATH domain matches.')
parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####
def read_data(uniprot_fetch):
    '''Read the uniprot fetch
    '''

    entry=[]
    name=[]
    length=[]
    uids=[]
    with open(uniprot_fetch, 'r') as file:
        ln=0
        for line in file:
            line = line.rstrip()
            line = line.split('\t')
            entry.append(line[0])
            name.append(line[1])
            length.append(line[2])
            uids.append(line[3])



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
uniprot_fetch = pd.read_csv(args.uniprot_fetch[0],sep='\t',low_memory=False)
outdir = args.outdir[0]

pdb.set_trace()
#Read data
read_data(uniprot_fetch)
#Analyze
analyze_distributions(df)
