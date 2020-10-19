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
                    'Y':18,'V':19,'X':20, '-':21
                  }