from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

import seq2seq

import prepare_data_for_model as p
import load_trim_data as d

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")



#searcher = seq2seq.GreedySearchDecoder(r.encoder, r.decoder)
#seq2seq.evaluateInput(r.encoder, r.decoder, searcher, d.voc)


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = r"C:\Users\wxwyl\Desktop\nontopic-mostlikely\64000_checkpoint.tar"
checkpoint_iter = 64000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    d.voc.__dict__ = checkpoint['voc_dict']
    #print("not empty")


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(d.voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = seq2seq.EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = seq2seq.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, d.voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')



encoder.eval()
decoder.eval()
searcher = seq2seq.GreedySearchDecoder(encoder, decoder)
seq2seq.evaluateInput(encoder, decoder, searcher, d.voc)

