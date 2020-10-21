import math
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
import numpy as np
import pdb


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """
    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def get_batch(self,grouped_labels,encoded_seqs,batch_size,s="train"):
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

    def generate(self,grouped_labels,encoded_seqs,batch_size, s="train"):
        """
        a generator for batches, so model.fit_generator can be used.
        """
        while True:
            pairs, targets = self.get_batch(grouped_labels,encoded_seqs,batch_size,s)
            yield (pairs, targets)

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)
        print(' ',lr)

    def find(self, grouped_labels,encoded_seqs, start_lr, end_lr, batch_size=64, epochs=1):
        num_batches = epochs * len(grouped_labels) / batch_size
        self.lr_mult = (end_lr / start_lr) ** (1 / num_batches)

        # Save weights into a file
        self.model.save_weights('tmp.h5')

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit_generator(self.generate(grouped_labels,encoded_seqs,batch_size),
                steps_per_epoch=int(len(grouped_labels)/batch_size), epochs=epochs,
                        callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights('tmp.h5')

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)
