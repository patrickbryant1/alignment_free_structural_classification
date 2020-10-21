#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Neural Network for embedding structural space.''')

parser.add_argument('--lrs_losses', nargs=1, type= str, default=sys.stdin, help = 'lr and losses.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####
def plot_lrfinder(lrs_losses):
    '''Plot the lrs and losses
    '''
    plt.plot(lrs_losses[0,:],lrs_losses[1,:]) #Plot lrs vs losses
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig(outdir+'lr_vis.png', format='png', dpi=300)

######################MAIN######################
args = parser.parse_args()
lrs_losses = np.load(args.lrs_losses[0], allow_pickle=True)
outdir = args.outdir[0]
pdb.set_trace()
#Plot
plot_lrfinder(lrs_losses)
