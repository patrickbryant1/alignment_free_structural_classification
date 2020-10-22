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
def check_order_variation(unique_domain_combos):
    '''Check that the combos are unique by analyzing the order of domains and possible variations of this
    Domains in a protein with order AB are treated equal as those with order BA,
    however order ABC is not considered to be equal to order CBA
    '''
    #Split the combos into parts
    combos = np.array(unique_domain_combos['Cross-reference (Gene3D)'])
    unique_combos = {} #Save the unique combos
    for c in range(len(combos)):
        combo = combos[c].split(';')[:-1]
        if len(combo)==2:
            #Check if the combo or a variation of it is in unique combos
            found = False #Keep track of if any combo of the current domain set exists in unique_combos
            for i in range(len(combo)):
                p1 = ';'.join(combo[i:])
                p2 = ';'.join(combo[:i])
                key = p1+p2+';'
                if key in [*unique_combos.keys()]:
                    found = True
            #Check if combo already found
            if found == True:
                continue
            else:
                unique_combos[combos[c]]=c
        else:
            unique_combos[combos[c]]=c

    return unique_combos



#####MAIN#####
args = parser.parse_args()
uniprot_fetch = pd.read_csv(args.uniprot_fetch[0],sep='\t',low_memory=False)
outdir = args.outdir[0]

#Look at how many lack domain annotations
num_lacking_domain = len(uniprot_fetch['Cross-reference (Gene3D)'])-len(uniprot_fetch['Cross-reference (Gene3D)'].dropna()) #No nans
print('Number of entries lacking domain annotations', num_lacking_domain)
#Unique domain combos - NOTE! Need to make sure there are not combos of varying order as well
unique_domain_combos = uniprot_fetch.drop_duplicates(subset=['Cross-reference (Gene3D)'])
unique_domain_combos = unique_domain_combos.dropna()
unique_domain_combos = unique_domain_combos.reset_index()

#Get a selection of unique domain combinations, where each combination is represented only once
sel_unique_domain_combos = check_order_variation(unique_domain_combos)
print('There are', len(sel_unique_domain_combos.keys()), 'unique domain combinations in the', len(uniprot_fetch), 'sequences.')
print('Removed', len(unique_domain_combos)-len(sel_unique_domain_combos.keys()), 'by order analysis')

#Select the unique ones from the df
unique_combo_df = unique_domain_combos.loc[[*sel_unique_domain_combos.values()]]
unique_combo_df = unique_combo_df.reset_index()
#Save the df
unique_combo_df.to_csv(outdir+'unique_domain_combinations.csv')
