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
        pdb.set_trace()
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
names = []
all_losses = []
for name in losses:
	names.append(name.split('/')[-2])
	all_losses.append(np.load(name, allow_pickle=True))
all_losses = np.array(all_losses)
np.save(outdir+'names.npy',np.array(names))
np.save(outdir+'all_losses.npy',all_losses)
pdb.set_trace()	
plot_losses(losses)
