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
    try:
        pred_ranks = np.load(outdir+'pred_ranks.npy', allow_pickle=True)
    except:
        pred_ranks = []
        pred_min2seqs_ranks = []
        n_with_one=0
        for i in range(len(grouped_labels)):
            group = grouped_labels[i] #Group
            print(i)
            if len(group)<2:
                pred_ranks.append(0)
                n_with_one+=1
                continue
            for j in group:
                #Compute the cosine similarity between the individual embeddings and all group embeddings
                #It turns out, the closer the documents are by angle, the higher is the Cosine Similarity

                sims = cosine_similarity(np.array([embeddings[j]]),class_embeddings)[0]

                ranks = np.argsort(sims)
                #Reverse ranks
                ranks = ranks[::-1] #Since the cosine sim should be maximized, the ranks are reversed.

                pred_ranks.append(np.where(ranks==i)[0][0])
                pred_min2seqs_ranks.append(np.where(ranks==i)[0][0])
                #nonj = np.setdiff1d(group,j)
                #group_ranks = []
                #Go through all ranks
                #for k in nonj:
                #    group_ranks.append(np.where(ranks==k)[0][0])
                #Save the best rank for the group
                #pred_ranks.append(min(group_ranks))


        #Save predicted_ranks
        pred_ranks = np.asarray(pred_ranks)


        np.save(outdir+'pred_ranks.npy', pred_ranks)
        print('Average rank',np.average(pred_min2seqs_ranks))
        print('There are', n_with_one, 'H-groups with only one sequence')

    #Investigate num above threshold
    num_above_t = []
    fpr = []
    for i in range(0,len(class_embeddings),10):
        if i ==0:
            i=1
        num_above_t.append(np.where(pred_ranks<i)[0].shape[0])
        #FPR
        #When top 10 are called, 9 FP are called for every called TP
        fpr.append((i-1)/(i-1+len(class_embeddings)-i)) #The fpr will be the threshold -1 divided by the number of classes
                                                #e.g. on top 10, 9 are FP out of all the negatives = FP+TN = 9 + num_classes-t
                                                #This means the FPR will be 10-1/(10-1+num_classes-10)=9/(num_classes-1) (only one class is positive from the beginning)

    num_above_t = np.array(num_above_t)
    plt.plot(100*np.array(fpr),100*num_above_t/len(embeddings))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(outdir+'topn.png', format='png', dpi=300)


    #false_positives = num_above_t- #Num above t that should not be divided by all negative
    top1 = np.where(pred_ranks<1)[0].shape[0]
    top10 = np.where(pred_ranks<10)[0].shape[0]
    top100 = np.where(pred_ranks<100)[0].shape[0]
    top1000 = np.where(pred_ranks<1000)[0].shape[0]
    print('There are', top1, 'sequences ranked top1. Equaling', np.round(100*top1/len(embeddings),2),'%')
    print('There are', top10, 'sequences ranked top10. Equaling', np.round(100*top10/len(embeddings),2),'%')
    print('There are', top100, 'sequences ranked top100. Equaling', np.round(100*top100/len(embeddings),2),'%')
    print('There are', top1000, 'sequences ranked top1000. Equaling', np.round(100*top1000/len(embeddings),2),'%')
    pdb.set_trace()
    return None



######################MAIN######################
args = parser.parse_args()

t1 = time.time()
sequence_df = pd.read_csv(args.sequence_df[0])
json_file = (args.json_file[0])
weights = (args.weights[0])
outdir = args.outdir[0]
#Assign data and labels
#Onehot encode sequences
sequences = np.array(sequence_df['sequence'])
encoded_seqs = one_hot(sequences)

#Get H-group labels
try:
    grouped_labels = np.load(outdir+'grouped_labels_s100.npy', allow_pickle=True)
except:
    hgroup_labels = np.array(sequence_df['H-group'])
    grouped_labels = group_by_hgroup(hgroup_labels)
    np.save(outdir+'grouped_labels_s100.npy', grouped_labels) #Save

t2 = time.time()
print('Formatted in', np.round(t2-t1,2),'seconds')



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
    batch_size=64
    #Get average embeddings for all entries

    embeddings = np.zeros((len(encoded_seqs),10))
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
