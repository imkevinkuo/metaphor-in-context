import random

from util import get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix
from util import evaluate, predict
from model import RNNSequenceModel

import torch
import h5py
import ast
import data_parser

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True

# These affect embedding behavior
torch.random.manual_seed(0)
torch.cuda.manual_seed_all(0)

"""
1. Data pre-processing
"""
'''
1.1 VUA
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a list of labels: 
    a list of pos: 

'''
raw_train_toefl = data_parser.load_raw_train_toefl()
raw_test_toefl = data_parser.load_raw_test_toefl()
pos_set = data_parser.get_pos_set(raw_train_toefl, raw_test_toefl)

# embed the pos tags
pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)

for i in range(len(raw_train_toefl)):
    raw_train_toefl[i][2] = index_sequence(pos2idx, raw_train_toefl[i][2])

for i in range(len(raw_test_toefl)):
    raw_test_toefl[i][1] = index_sequence(pos2idx, raw_test_toefl[i][1])

print('size of training, test set: ', len(raw_train_toefl), len(raw_test_toefl))

"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_train_toefl + raw_test_toefl)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
elmos_train_toefl = h5py.File('../elmo/TOEFL_train.hdf5', 'r')
elmos_test_toefl = h5py.File('../elmo/TOEFL_test.hdf5', 'r')
# no suffix embeddings for sequence labeling
suffix_embeddings = None

'''
2. 2
embed the datasets
'''
random.seed(0)
random.shuffle(raw_train_toefl)

sentence_to_index_train = ast.literal_eval(elmos_train_toefl['sentence_to_index'][0])
labels = [example[1] for example in raw_train_toefl]
poss = [example[2] for example in raw_train_toefl]
sentences = [embed_indexed_sequence(example[0], example[2], word2idx, glove_embeddings,
                                    elmos_train_toefl, suffix_embeddings, sentence_to_index_train[example[0]])
             for example in raw_train_toefl]


# Test set
sentence_to_index_test = ast.literal_eval(elmos_test_toefl['sentence_to_index'][0])
embedded_test_toefl = [[embed_indexed_sequence(example[0], example[1], word2idx, glove_embeddings,
                                               elmos_test_toefl, suffix_embeddings, sentence_to_index_test[example[0]]),
                        example[1], example[2]] for example in raw_test_toefl]


def test_model(RNNseq_model, loss_criterion, val_dataloader):
    avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader, RNNseq_model, loss_criterion, using_GPU)
    return performance_matrix, pos2idx
