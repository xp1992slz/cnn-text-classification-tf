import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Dataset Names
MR = "MR"
EC = "EC"
TW = "TW"

def load_data_and_labels(dataset_name):
    """
    Dataset name
    1. MR - Rotten Tomatoes Movie Review Data (Positive/Negative)
    2. EC - Emotional Cause Data (Happiness/Sadness/Anger/Fear/Surprise/Disgust/Shame)
    3. TW - Twitter data
    """

    if dataset_name == MR:
        # Load data from files
        positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]
    elif dataset_name == EC:
        # Load from EC
        data = list(open("./data/emotion_cause/No Cause refine.txt", "r").readlines())
        data = [s.split('\t') for s in data]
        x_text = []
        y = []
        for s in data:
            x_text.append(clean_str(s[1].strip()))
            y.append(int(s[0]))

        number_of_classes = max(y)+1
        y_one_hot = []
        for i in y:
            l = [0] * number_of_classes
            l[i] = 1
            y_one_hot.append(l)
        return [x_text, np.reshape(y_one_hot, (len(y_one_hot), number_of_classes))]
    elif dataset_name == TW:
        pos_iter = pd.read_csv("./data/twitter/training_pos.csv", iterator=True, chunksize=10000, engine='python', sep='delimiter', header=None)
        neg_iter = pd.read_csv("./data/twitter/training_neg.csv", iterator=True, chunksize=10000, engine='python', sep='delimiter', header=None)

        #positive_examples = list(open("./data/twitter/training_pos.csv", "r").readlines())
        positive_examples = pd.concat(pos_iter, ignore_index=True)
        print "Finished loading the pos csv files"
        positive_examples = positive_examples[0].str.replace(r"[^A-Za-z0-9(),!?\'\`]", " ").str.replace(r"\'s", " \'s").str.replace(r"\'ve", " \'ve").str.replace(r"n\'t", " n\'t").str.replace(r"\'re", " \'re").str.replace(r"\'d", " \'d").str.replace(r"\'ll", " \'ll").str.replace(r",", " , ").str.replace(r"!", " ! ").str.replace(r"\(", " \( ").str.replace(r"\)", " \) ").str.replace(r"\?", " \? ").str.replace(r"\s{2,}", " ").str.strip().str.lower()
        print "Finished cleaning the pos csv files"
        print (positive_examples[:-10])

        negative_examples = pd.concat(neg_iter, ignore_index=True)
        print "Finished loading the neg csv files"
        negative_examples = negative_examples[0].str.replace(r"[^A-Za-z0-9(),!?\'\`]", " ").str.replace(r"\'s", " \'s").str.replace(r"\'ve", " \'ve").str.replace(r"n\'t", " n\'t").str.replace(r"\'re", " \'re").str.replace(r"\'d", " \'d").str.replace(r"\'ll", " \'ll").str.replace(r",", " , ").str.replace(r"!", " ! ").str.replace(r"\(", " \( ").str.replace(r"\)", " \) ").str.replace(r"\?", " \? ").str.replace(r"\s{2,}", " ").str.strip().str.lower()
        print "Finished cleaning the neg csv files"
        print (negative_examples[:-10])

        # Split by words
        x_text = positive_examples + negative_examples
        x_text = np.array(x_text)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        y = np.array(y)

        print "Finished Generating Labels"

        shuffle_indices = np.random.permutation(np.arange(len(x_text)))
        shuffled_data = x_text[shuffle_indices]
        shuffled_labels = y[shuffle_indices]

        print "Finished Shuffling"

        return [shuffled_data[:len(x_text)], shuffled_labels[:len(x_text)]]

    raise ValueError('Wrong Data Set Name')

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
