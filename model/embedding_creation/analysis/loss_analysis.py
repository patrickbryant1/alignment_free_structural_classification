#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Neural Network for embedding structural space.''')

parser.add_argument('--indir', nargs=1, type= str, default=sys.stdin, help = 'loss per epoch.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####
def plot_losses(losses):
    '''Plot the lrs and losses
    '''
    for name in losses:
        name_losses = np.load(name, allow_pickle=True)
        plt.plot(np.arange(len(name_losses)),name_losses) #Plot losses
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(outdir+'loss_vis.png', format='png', dpi=300)

######################MAIN######################
args = parser.parse_args()
indir = args.indir[0]
outdir = args.outdir[0]

#Get all losses
losses = glob.glob(indir+'*/losses.npy')
pdb.set_trace()
plot_losses(losses)
