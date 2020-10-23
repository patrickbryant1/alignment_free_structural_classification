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
    #Params
    batch_size=[]
    filters = []
    num_res_blocks=[]
    dilation_rate = []
    step_size=[]
    num_cycles=[]

    min_loss = []#Save min loss
    fig,ax = plt.subplots(figsize=(9/2.54, 6/2.54))
    for i in range(len(all_losses)):
        plt.plot(np.arange(len(all_losses[i])),all_losses[i], linewidth=0.5) #Plot losses
        #Min loss
        min_loss.append(min(all_losses[i]))
        #Get params
        name = all_names[i].split('_')
        batch_size.append(name[0])
        filters.append(name[1])
        num_res_blocks.append(name[2])
        dilation_rate.append(name[3])
        step_size.append(name[4])
        num_cycles.append(name[5])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #Min loss
    min_combo = np.where(min_loss ==min(min_loss))[0][0]
    plt.text(400,min_loss[min_combo],all_names[min_combo])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plt.savefig(outdir+'loss_vis.png', format='png', dpi=300)
    plt.close()
    #Plot loss distribution
    fig,ax = plt.subplots(figsize=(9/2.54, 6/2.54))
    sns.distplot(min_loss)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Min loss')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(outdir+'loss_distr.png', format='png', dpi=300)

    #Analyze the relationship btw the parameters and the loss
    pdb.set_trace()
######################MAIN######################
#Plt
plt.rcParams.update({'font.size': 7})
#Parse args
args = parser.parse_args()
indir = args.indir[0]
outdir = args.outdir[0]

#Get all losses
try:
    all_losses = np.load(outdir+'all_losses.npy', allow_pickle=True)
    all_names = np.load(outdir+'names.npy', allow_pickle=True)
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
