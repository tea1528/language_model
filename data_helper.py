import math
import json
import os
import numpy as np

MAX_SEQ_LEN = 10


def read_data_from_file(filename):
    sentence = []
    with open(filename, "r") as fin:
        fin.readline()
        for line in fin:
            _, prev_sent, last_token = line.strip().split(",")
            sentence += prev_sent.split() + [last_token, "<eos>"]
    return sentence


def build_input_data(sentence, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    unknown_token_id = vocabulary["<unk>"]
    vocab = vocabulary.keys()
    sentence_id = [vocabulary[word] if word in vocab else unknown_token_id for word in sentence]

    x = []
    y = []

    num_sentences = math.ceil(len(sentence_id) / MAX_SEQ_LEN )
    '''
    The the len of last sentence may be less than MAX_SEQ_LEN, so we pad it using tokens in the begining.
    corpus: a cat sits on the mat
    x(text): a cat sits
    y(text): cat sits on
    '''
    sentence_id += sentence_id[:MAX_SEQ_LEN + 1]
    while len(sentence_id) < MAX_SEQ_LEN:
        sentence_id += sentence_id[:MAX_SEQ_LEN+1]

    for i in range(num_sentences):
        x.append(sentence_id[i*MAX_SEQ_LEN:(i+1)*MAX_SEQ_LEN])
        y_ = sentence_id[i*MAX_SEQ_LEN+1:(i+1)*MAX_SEQ_LEN+1]
        y.append(y_)
    x = np.array(x)
    y = np.expand_dims(np.array(y), axis=-1)
    return x, y


def load_embedding_weights(vocabulary):
    embeddings_index = {}
    with open ('./data/glove.840B.300d.txt','r') as fin:
        for line in fin:
            values = line.strip().split(' ')
            word = values[0]
            emb = np.asarray(values[1:], dtype='float32')
            emb_dim = len(emb)
            embeddings_index[word] = emb

    embedding_matrix = np.zeros((len(vocabulary), emb_dim))
    for word in vocabulary.keys():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: # words not found in embedding index will be all-zeros.	
            embedding_matrix[vocabulary[word]] = embedding_vector

    return embedding_matrix


def load_data(data_path, debug=False):
    # get the data paths
    train_path = os.path.join(data_path, "valid.csv" if debug else "train.csv")
    valid_path = os.path.join(data_path, "valid.csv")
    vocab_path = os.path.join(data_path, "vocab.json")

    train_data = read_data_from_file(train_path)
    valid_data = read_data_from_file(valid_path)

    # build vocabulary from training data
    vocabulary = json.load(open(vocab_path))
    vocab_size = len(vocabulary)
    
    # get input data
    x_train, y_train = build_input_data(train_data, vocabulary)
    x_valid, y_valid = build_input_data(valid_data, vocabulary)

    return x_train, y_train, x_valid, y_valid, vocab_size


