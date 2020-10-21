#!/usr/share/python3
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


#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')

parser.add_argument('--json_file', nargs=1, type= str,
                  default=sys.stdin, help = 'path to .json file with keras model to be opened')

parser.add_argument('--weights', nargs=1, type= str,
                  default=sys.stdin, help = '''path to .h5 file containing weights for net.''')

parser.add_argument('--sequence_df', nargs=1, type= str, default=sys.stdin, help = 'Path to sequence_df.')

parser.add_argument('--outdir', nargs=1, type= str,
                  default=sys.stdin, help = '''path to output directory.''')


#Set random seed - use the same as wehen training the siamese net to be able to evaluate the GZSL vs SZL
np.random.seed(0)

######################FUNCTIONS######################


def load_model(json_file, weights):

	global model

	json_file = open(json_file, 'r')
	model_json = json_file.read()
	model = model_from_json(model_json)
	model.load_weights(weights)
	model._make_predict_function()
	#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

######################MAIN######################
args = parser.parse_args()
t1 = time.time()
sequence_df = pd.read_csv(args.sequence_df[0])

#Assign data and labels
#Onehot encode sequences
sequences = np.array(sequence_df['sequence'])
encoded_seqs = one_hot(sequences)

#Get H-group labels
try:
    grouped_labels = np.load(outdir+'grouped_labels.npy', allow_pickle=True)
except:
    hgroup_labels = np.array(sequence_df['H-group'])
    grouped_labels = group_by_hgroup(hgroup_labels)

t2 = time.time()
print('Formatted in', np.round(t2-t1,2),'seconds')
args = parser.parse_args()
json_file = (args.json_file[0])
weights = (args.weights[0])
outdir = args.outdir[0]


#Load and run model
model = load_model(json_file, weights)

#Get embedding layers
emb_layer = Model(inputs=model.input, outputs=model.get_layer('emb1').output)

batch_size=32
#Get average embeddings for all entries

for i in range(0,len(sequences)-batch_size,batch_size):
    encoded_seqs = [] #Encoded sequences
    for j in range(i,i+batch_size):
        encoded_seqs.append(np.eye(21)[sequences[i]])
    #Obtain embeddings
    embeddings = np.asarray(emb_layer.predict([X]))

average_emb = np.average([emb1, emb2], axis = 0)
#Save embeddings
#np.save(out_dir+'average_feature_emb.npy', average_emb)

#Compute class labels by averaging the embeddings for each H-group.
class_embeddings = []
unique_groups = np.unique(y)

for i in range(0,len(unique_groups)):
    emb_match = average_emb[np.where(y == unique_groups[i])]
    class_emb = np.average(emb_match, axis = 0)
    class_embeddings.append(class_emb)

class_embeddings = np.asarray(class_embeddings)
#Save class embeddings
np.save(out_dir+'class_emb.npy', class_embeddings)

train_index = np.isin(y, converted_train_classes)
test_index = np.isin(y, test_classes)

def alternative_sim(class_embeddings, emb):
    '''Compute an alternative similarity measure
    '''
    diff_norm = np.linalg.norm(class_embeddings-emb, axis = 1)
    class_emb_norm = np.linalg.norm(class_embeddings, axis = 1)
    emb_norm = np.linalg.norm(emb)

    esim = 1/(1+2*diff_norm/(class_emb_norm+emb_norm))
    return esim

def zsl_test(indices, type, out_dir):
    "A function that runs ZSL according to provided data"
    item = average_emb[indices]
    targets = y[indices]
    name = 'average_emb'

    #Compute L1 distance to all class_embeddings
    pred_ranks = []
    for i in range(0, len(item)):
        true = targets[i]
        #diff = np.absolute(class_embeddings-item[i])
        #dists = np.sum(diff, axis = 1)
        dists =  alternative_sim(class_embeddings, item[i])
        ranks = np.argsort(dists)
        try:
            rank = np.where(ranks == true)[0][0]
        except:
            pdb.set_trace()

        pred_ranks.append(rank)

    #Save predicted_ranks
    pred_ranks = np.asarray(pred_ranks)
    np.save(out_dir+type+'_'+name+'_pred_ranks.npy', pred_ranks)

    return None

zsl_test(train_index, 'train', out_dir)
zsl_test(test_index, 'test', out_dir)
pdb.set_trace()
