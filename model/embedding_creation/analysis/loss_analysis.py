#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Neural Network for embedding structural space.''')

parser.add_argument('--indir', nargs=1, type= str, default=sys.stdin, help = 'loss per epoch.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')


#####FUNCTIONS#####
def plot_losses(all_losses, all_names):
    '''Plot the epochs and losses
    '''
    min_loss = []
    for loss in all_losses:
        plt.plot(np.arange(len(loss)),loss) #Plot losses
        min_loss.append(min(loss))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(outdir+'loss_vis.png', format='png', dpi=300)
    
    #Plot loss distribution

    pdb.set_trace()
######################MAIN######################
args = parser.parse_args()
indir = args.indir[0]
outdir = args.outdir[0]

#Get all losses
try:
    all_losses = np.load(outdir+'all_losses.npy', allow_pickle=True)
    all_losses = np.load(outdir+'names.npy', allow_pickle=True)
except:
    losses = glob.glob(indir+'*/losses.npy')
    names=[]
    all_losses = []
    for name in losses:
        names.append(name.split('/')[-2])
        all_losses.append(np.load(name, allow_pickle=True))
    all_losses = np.array(all_losses)
    np.save(outdir+'names.npy',np.array(names))
    np.save(outdir+'all_losses.npy',all_losses)
#Plot
plot_losses(all_losses, all_names)
