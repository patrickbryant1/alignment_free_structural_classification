#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../') #Makes it possible to use processing
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
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling1D,add,Lambda,Dense, Dropout, Activation, Conv1D, BatchNormalization, Flatten, Subtract
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

#Metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

#Vis
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pdb

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

def tsne_emb(embeddings, class_embeddings, sequence_df, outdir):
    '''Visualize the class embeddings and the individual embeddings. Color them by architecture.
    '''

    architectures = np.array(sequence_df['Class']) #Architectures
    u_architectures = sequence_df['Class'].unique() #Unique architectures
    colors = pl.cm.viridis(np.linspace(0,1,len(u_architectures)))
    #Perform t-SNE
    try:
        x = np.load(outdir+'tsne.npy', allow_pickle=True)
    except:
        x = TSNE(n_components=2).fit_transform(embeddings)
        np.save(outdir+'tsne.npy', x)
    #Color in each architecture
    for i in range(len(u_architectures)):
        arch = u_architectures[i]
        ind = np.where(architectures==arch)[0]
        sel = x[ind]
        plt.scatter(sel[:,0], sel[:,1], color=colors[i], s= 1, alpha = 0.5)

def alternative_sim(class_embeddings, emb):
    '''Compute an alternative similarity measure
    '''
    diff_norm = np.linalg.norm(class_embeddings-emb, axis = 1)
    class_emb_norm = np.linalg.norm(class_embeddings, axis = 1)
    emb_norm = np.linalg.norm(emb)

    esim = 1/(1+2*diff_norm/(class_emb_norm+emb_norm))
    return esim

def zsl(class_embeddings, embeddings, grouped_labels):
    '''A function that runs ZSL according to provided data
    '''

    pred_ranks = []
    for i in range(len(grouped_labels)):
        group = grouped_labels[i] #Group
        for j in group:
            #Compute the cosine similarity between the individual embeddings and all group embeddings
            #It turns out, the closer the documents are by angle, the higher is the Cosine Similarity

            sims = cosine_similarity(np.array([embeddings[j]]),embeddings)[0]

            ranks = np.argsort(sims)
            #Reverse ranks
            ranks = ranks[::-1] #Since the cosine sim should be maximized, the ranks are reversed.
            nonj = np.setdiff1d(group,j)
            group_ranks = []
            #Go through all ranks
            for k in nonj:
                group_ranks.append(np.where(ranks==k)[0][0])
            #Save the best rank for the group
            pred_ranks.append(min(group_ranks))

    pdb.set_trace()

    #Save predicted_ranks
    pred_ranks = np.asarray(pred_ranks)


    np.save(outdir+'_pred_ranks.npy', fixed_ranks)
    print('Average rank',np.average(fixed_ranks))

    return None



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

try:
    class_embeddings=np.load(outdir+'class_emb.npy', allow_pickle=True)
    embeddings=np.load(outdir+'emb.npy', allow_pickle=True)
except:
    #Load and run model
    model = load_model(json_file, weights)

    #Get embedding layers
    #emb_layer = Model(inputs=model.input, outputs=model.get_layer('emb1').output)
    inp = model.input                                           # input placeholder
    outputs = [model.layers[-2].output] #the second to last layer is the embedding, the last is the dense layer for classification
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]
    #K.function creates theano/tensorflow tensor functions which is later used to get the output from the symbolic graph given the input.
    #Now K.learning_phase() is required as an input as many Keras layers like Dropout/Batchnomalization depend on it to change behavior during training and test time
    #I use batch BatchNormalization
    batch_size=32
    #Get average embeddings for all entries

    embeddings = np.zeros((len(encoded_seqs),5))
    for i in range(0,len(encoded_seqs)-batch_size,batch_size):
        onehot_seqs = [] #Encoded sequences
        for j in range(i,i+batch_size):
            onehot_seqs.append(np.eye(21)[encoded_seqs[i]])
        #Obtain embeddings
        embeddings[i:i+batch_size]=functors[0](np.array(onehot_seqs))[0]

    #Compute class labels by averaging the embeddings for each H-group.
    class_embeddings = []

    for i in range(len(grouped_labels)):
        group_indices = grouped_labels[i]
        class_embeddings.append(np.median(embeddings[group_indices], axis = 0))

    class_embeddings = np.asarray(class_embeddings)
    #Save embeddings
    np.save(outdir+'emb.npy', embeddings)
    #Save class embeddings
    np.save(outdir+'class_emb.npy', class_embeddings)

#Look at embeddings
#tsne_emb(embeddings, class_embeddings, sequence_df, outdir)

#Run zero shot learning
zsl(class_embeddings, embeddings, grouped_labels)
pdb.set_trace()


zsl_test(train_index, 'train', out_dir)
zsl_test(test_index, 'test', out_dir)
pdb.set_trace()
