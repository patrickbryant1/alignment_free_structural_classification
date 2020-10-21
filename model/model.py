#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import pandas as pd
import time
#Preprocessing
from collections import Counter
from processing import one_hot, group_by_hgroup

#Keras
import tensorflow as tf
from tensorflow.keras import regularizers,optimizers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling1D,add,Lambda,Dense, Dropout, Activation, Conv1D, BatchNormalization, Flatten, Subtract
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
#visualization
from tensorflow.keras.callbacks import TensorBoard
from lr_finder import LRFinder

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Neural Network for embedding structural space.''')

parser.add_argument('--sequence_df', nargs=1, type= str, default=sys.stdin, help = 'Path to sequence_df.')

#parser.add_argument('--params_file', nargs=1, type= str, default=sys.stdin, help = 'Path to file with net parameters')

parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

#FUNCTIONS
def read_net_params(params_file):
    '''Read and return net parameters
    '''
    net_params = {} #Save information for net

    with open(params_file) as file:
        for line in file:
            line = line.rstrip() #Remove newlines
            line = line.split("=") #Split on "="

            net_params[line[0]] = line[1]


    return net_params




def get_batch(grouped_labels,encoded_seqs,batch_size,s="train"):
    """
    Create batch of n pairs
    """

    random_numbers = np.random.choice(len(grouped_labels),size=(batch_size,),replace=False) #without replacement

    #initialize vector for the targets
    targets=[]
    s1 = []
    #Get batch data - make half from the same H-group and half from different
    for i in random_numbers:
        matches = grouped_labels[i]
        pick = np.random.choice(matches,size=1, replace=False)
        s1.append(np.eye(21)[encoded_seqs[pick[0]]])
        targets.append(np.eye(len(grouped_labels))[i])

    return np.array(s1), np.array(targets)

def generate(grouped_labels,encoded_seqs,batch_size, s="train"):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        pairs, targets = get_batch(grouped_labels,encoded_seqs,batch_size,s)
        yield (pairs, targets)

######################MAIN######################
args = parser.parse_args()
t1 = time.time()
sequence_df = pd.read_csv(args.sequence_df[0])
#params_file = args.params_file[0]
outdir = args.outdir[0]

#Assign data and labels
np.random.seed(2) #Set random seed - ensures same split every time

#Onehot encode sequences
sequences = np.array(sequence_df['sequence'])
encoded_seqs = one_hot(sequences)

#Too large to save


#Get H-group labels
try:
    grouped_labels = np.load(outdir+'grouped_labels.npy', allow_pickle=True)
except:
    hgroup_labels = np.array(sequence_df['H-group'])
    grouped_labels = group_by_hgroup(hgroup_labels)
    #Save
    np.save(outdir+'grouped_labels.npy',grouped_labels)
t2 = time.time()
print('Formatted in', np.round(t2-t1,2),'seconds')

#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=outdir+log_name)

######MODEL######
#Parameters
#net_params = read_net_params(params_file)

#Variable params
num_epochs=200
batch_size = 64 #int(net_params['batch_size'])
kernel_size = 21
filters =20
input_dim = (600,21)
num_res_blocks=1
dilation_rate = 5
seq_length=600

#lr opt
find_lr = True
#LR schedule
step_size = 10
num_cycles = 10
num_epochs = step_size*2*num_cycles
num_steps = int(len(grouped_labels)/batch_size)
max_lr = 0.0015
min_lr = max_lr/10
lr_change = (max_lr-min_lr)/step_size  #(step_size*num_steps) #How mauch to change each batch
lrate = min_lr

#####MODEL#####
#####Inputs#####
in_1 = keras.Input(shape = [600,21])


def resnet(x, num_res_blocks):
	"""Builds a resnet with 1D convolutions of the defined depth.
	"""


	# Instantiate the stack of residual units
	for res_block in range(num_res_blocks):
		batch_out1 = BatchNormalization()(x) #Bacth normalize, focus on segment
		activation1 = Activation('relu')(batch_out1)
		conv_out1 = Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, input_shape=input_dim, padding ="same")(activation1)
		batch_out2 = BatchNormalization()(conv_out1) #Bacth normalize, focus on segment
		activation2 = Activation('relu')(batch_out2)
        #Downsample - half filters
		conv_out2 = Conv1D(filters = int(filters/2), kernel_size = kernel_size, dilation_rate = 1, input_shape=input_dim, padding ="same")(activation2)
		x = Conv1D(filters = int(filters/2), kernel_size = kernel_size, dilation_rate = 1, input_shape=input_dim, padding ="same")(x)
		x = add([x, conv_out2]) #Skip connection

	return x

#Initial convolution
in_1_conv = Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = 2, input_shape=input_dim, padding ="same")(in_1)
#Output (batch, steps(len), filters), filters = channels in next
x1 = resnet(in_1_conv, num_res_blocks)


#Maxpool along sequence axis
maxpool1 = MaxPooling1D(pool_size=seq_length)(x1)

flat1 = Flatten()(maxpool1)  #Flatten

probabilities = Dense(len(grouped_labels), activation='softmax')(flat1)

#Checkpoint
#filepath=outdir+"weights-{epoch:02d}-.hdf5"
#checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max')

#Model: define inputs and outputs
model = Model(inputs = in_1, outputs = probabilities)
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics = ['accuracy'])

if find_lr == True:
    lr_finder = LRFinder(model)
    #Encode all data
    lr_finder.find(grouped_labels,encoded_seqs, start_lr=0.00001, end_lr=1, batch_size=batch_size, epochs=1)
    losses = lr_finder.losses
    lrs = lr_finder.lrs
    l_l = np.asarray([lrs, losses])
    np.savetxt(outdir+'lrs_losses.txt', l_l)
    num_epochs = 0

#LR schedule
class LRschedule(Callback):
  '''lr scheduel according to one-cycle policy.
  '''
  def __init__(self, interval=1):
    super(Callback, self).__init__()
    self.lr_change = lr_change #How mauch to change each batch
    self.lr = min_lr
    self.interval = interval

  def on_epoch_end(self, epoch, logs={}):
    if epoch > 0 and epoch%step_size == 0:
      self.lr_change = self.lr_change*-1 #Change decrease/increase

    self.lr = self.lr + self.lr_change
    keras.backend.set_value(self.model.optimizer.lr, self.lr)


#Lrate
lrate = LRschedule()


#Summary of model
print(model.summary())

#Callbacks
callbacks=[lrate, tensorboard]
#Fit model
#Should shuffle uid1 and uid2 in X[0] vs X[1]
model.fit_generator(generate(grouped_labels,encoded_seqs,batch_size),
            steps_per_epoch=num_steps,
            epochs=num_epochs,
            callbacks=callbacks
            )
