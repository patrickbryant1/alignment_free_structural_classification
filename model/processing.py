#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''Functions for processing sequence data
'''

import argparse
import sys
import numpy as np
import pdb



#####FUNCTIONS#####

def one_hot(sequences):
    '''Onehot encode sequence data
    '''
    AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                    'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                    'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                    'Y':18,'V':19,'X':20
                  }

    encoded_seqs = []
    #Loop through all sequences
    for seq in sequences:
        enc_seq = []
        for aa in seq: #Go through all the amino acids in the sequence
            enc_seq.append(AMINO_ACIDS[aa])
        #pad/cut

        if len(seq)>600:
            enc_seq = enc_seq[:600]
        else:
            zeros = np.zeros(600,dtype='int32')
            zeros[:len(seq)]=enc_seq
            enc_seq=zeros

        #Save the sncoded sequence
        encoded_seqs.append(np.eye(21)[enc_seq]) #Onehot

    return np.array(encoded_seqs)


def group_by_hgroup(hgroup_labels):
    '''Group the sequences by H-group
    '''

    unique_hgroups = np.unique(hgroup_labels)
    match_indices = [] #Indices for sequences belonging to the same H-group
    #Go through all unique hgroups
    for u_group in unique_hgroups:
        match_indices.append(np.where(hgroup_labels==u_group)[0])

    return np.array(match_indices)
