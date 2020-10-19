#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
from ast import literal_eval
import pandas as pd
import glob

#Preprocessing
from collections import Counter
from processing import one_hot

#Keras
import tensorflow as tf
from tensorflow.keras import regularizers,optimizers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, Flatten, Subtract
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
#visualization
from tensorflow.keras.callbacks import TensorBoard

#from lr_finder import LRFinder

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


def get_batch(sequences,batch_size,s="train"):
    """
    Create batch of n pairs
    """

    random_numbers = np.random.choice(len(sequences),size=(batch_size,),replace=False) #without replacement

    #initialize vector for the targets
    targets=[]

    #Get batch data - make half from the same H-group and half from different
    for i in random_numbers:


    return pairs, targets

def generate(sequences,batch_size, s="train"):
    """
    a generator for batches, so model.fit_generator can be used.
    """
    while True:
        pairs, targets = get_batch(sequences,batch_size,s)
        yield (pairs, targets)

######################MAIN######################
args = parser.parse_args()
sequence_df = pd.read_csv(args.sequence_df[0])
#params_file = args.params_file[0]
outdir = args.outdir[0]

#Assign data and labels
np.random.seed(2) #Set random seed - ensures same split every time

#Onehot encode sequences
sequences = np.array(sequence_df['sequence'])
encoded_seqs = one_hot(sequences)
#Get H-group labels

#Tensorboard for logging and visualization
#log_name = str(time.time())
#tensorboard = TensorBoard(log_dir=out_dir+log_name)


######MODEL######
#Parameters
net_params = read_net_params(params_file)

#Variable params

batch_size = 32 #int(net_params['batch_size'])
#MODEL
in_1 = keras.Input(shape = [None,21])
in_2 = keras.Input(shape = [None,21])

#Initial convolution
in_1_conv = LSTM(10, return_sequences=True)(in_1)
in_2_conv = LSTM(10, return_sequences=True)(in_1)

x1 = resnet(in_1_conv, num_res_blocks)
x2 = resnet(in_2_conv, num_res_blocks)

act1=Dense(10, activation='softmax')(x1)
act2=Dense(10, activation='softmax')(x2)

# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([act1, act2])


probabilities = Dense(2, activation='softmax')(L1_distance)

#Checkpoint
#filepath=out_dir+"weights-{epoch:02d}-.hdf5"
#checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max')

#Model: define inputs and outputs
model = Model(inputs = [in_1, in_2], outputs = probabilities)
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss='binary_crossentropy',
              optimizer=opt)

#Summary of model
print(model.summary())

#Fit model
#Should shuffle uid1 and uid2 in X[0] vs X[1]
model.fit_generator(generate(sequences,batch_size),
            steps_per_epoch=int(2*len(sequences)/batch_size),
            epochs=num_epochs,
            shuffle=True #Dont feed continuously
            )
