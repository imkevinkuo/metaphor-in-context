import ast
import math
import numpy as np
import random

from tqdm import tqdm

from util import get_num_lines, get_vocab, embed_sequence, get_word2idx_idx2word, get_embedding_matrix
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate, predict
from model import RNNSequenceClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import h5py
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
    a index: int: idx of the focus verb
    a label: int 1 or 0

'''
raw_train_toefl = data_parser.load_raw_train_toefl()[0]
raw_test_toefl = data_parser.load_raw_test_toefl()[0]

print('TOEFL dataset division: ', len(raw_train_toefl), len(raw_test_toefl))

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
# suffix_embeddings: number of suffix tag is 2, and the suffix embedding dimension is 50
suffix_embeddings = nn.Embedding(2, 50)

'''
2. 2
embed the datasets
'''
random.seed(0)
random.shuffle(raw_train_toefl)

sentence_to_index_train = ast.literal_eval(elmos_train_toefl['sentence_to_index'][0])
sentences = [embed_sequence(example[0], example[1], word2idx, glove_embeddings, elmos_train_toefl,
                            suffix_embeddings, sentence_to_index_train[example[0]])
             for example in raw_train_toefl]
labels = [example[2] for example in raw_train_toefl]

assert(len(sentences) == len(labels))

# Test set
elmos_test_toefl = h5py.File('../elmo/TOEFL_test.hdf5', 'r')
sentence_to_index_test = ast.literal_eval(elmos_test_toefl['sentence_to_index'][0])
embedded_test_toefl = [[embed_sequence(example[0], example[1], word2idx, glove_embeddings, elmos_test_toefl,
                                       suffix_embeddings, sentence_to_index_test[example[0]]), example[2]]
                       for example in raw_test_toefl]


def train_k_fold(k):
    '''
    2. 3
    set up Dataloader for batching
    '''
    fold_size = int(len(raw_train_toefl) / 10)
    for i in range(k):
        val_indices = [z for z in range(i * fold_size, (i + 1) * fold_size)]
        embedded_train_toefl = [[sentences[j], labels[j]] for j in range(len(sentences)) if j not in val_indices]
        embedded_val_toefl = [[sentences[j], labels[j]] for j in range(len(sentences)) if j in val_indices]

        train_dataset_toefl = TextDataset([example[0] for example in embedded_train_toefl],
                                          [example[1] for example in embedded_train_toefl])
        val_dataset_toefl = TextDataset([example[0] for example in embedded_val_toefl],
                                        [example[1] for example in embedded_val_toefl])

        # Data-related hyperparameters
        batch_size = 64
        # Set up a DataLoader for the training, validation, and test dataset
        train_dataloader_toefl = DataLoader(dataset=train_dataset_toefl, batch_size=batch_size, shuffle=True,
                                            collate_fn=TextDataset.collate_fn)
        val_dataloader_toefl = DataLoader(dataset=val_dataset_toefl, batch_size=batch_size,
                                          collate_fn=TextDataset.collate_fn)

        clf, crit = train_model(train_dataloader_toefl, val_dataloader_toefl, i)


def train_model(train_dataloader_toefl, val_dataloader_toefl, fold_num):
    rnn_clf = RNNSequenceClassifier(num_classes=2, embedding_dim=300 + 1024 + 50, hidden_size=300, num_layers=1,
                                    bidir=True, dropout1=0.3, dropout2=0.2, dropout3=0.2)
    # Move the model to the GPU if available
    if using_GPU:
        rnn_clf = rnn_clf.cuda()
    # Set up criterion for calculating loss
    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    rnn_clf_optimizer = optim.SGD(rnn_clf.parameters(), lr=0.01, momentum=0.9)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 50

    '''
    3. 2
    train model
    '''
    training_loss = []
    val_loss = []
    training_f1 = []
    val_f1 = []
    # A counter for the number of gradient updates
    num_iter = 0
    for epoch in tqdm(range(num_epochs)):
        # print("Starting epoch {}".format(epoch + 1))
        for (example_text, example_lengths, example_labels) in train_dataloader_toefl:
            example_text = Variable(example_text)
            example_lengths = Variable(example_lengths)
            example_labels = Variable(labels)
            if using_GPU:
                example_text = example_text.cuda()
                example_lengths = example_lengths.cuda()
                example_labels = example_labels.cuda()
            # predicted shape: (batch_size, 2)
            predicted = rnn_clf(example_text, example_lengths)
            batch_loss = nll_criterion(predicted, example_labels)
            rnn_clf_optimizer.zero_grad()
            batch_loss.backward()
            rnn_clf_optimizer.step()
            num_iter += 1
            # Calculate validation and training set loss and accuracy every 200 gradient updates
            if num_iter % 200 == 0:
                precision, recall, f1, eval_accuracy, fus_f1 = test_model(rnn_clf, nll_criterion, val_dataloader_toefl)
                print(
                    "Iteration {}. P {}, R {}, A {}, F1 {}, MaF1 {}.".format(
                        num_iter, precision, recall, eval_accuracy, f1, fus_f1))
                filename = f'../models/classification/TOEFL_fold_{fold_num}_iter_{num_iter}.pt'
                torch.save(rnn_clf.state_dict(), filename)
    return rnn_clf, nll_criterion


def test_model(rnn_clf, nll_criterion, val_dataloader_toefl):
    avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(val_dataloader_toefl, rnn_clf,
                                                                           nll_criterion, using_GPU)
    return precision, recall, f1, eval_accuracy.item(), fus_f1


def predict_toefl(rnn_clf):
    preds = {}
    for (embed, txt_sent_id) in embedded_test_toefl:
        ex_data = TextDataset([embed], [0])
        ex_dataloader = DataLoader(dataset=ex_data, batch_size=1, collate_fn=TextDataset.collate_fn)
        pred = predict(ex_dataloader, rnn_clf, using_GPU)
        preds[txt_sent_id] = pred.item()
    return preds


def write_predictions_to_answer_file(predictions):
    import data_parser
    vtoks = data_parser.load_toefl_ptoks()
    answers = []
    for vtok in vtoks:
        answers.append(f"{vtok},{predictions[vtok]}\n")

    with open("answer.txt", "w") as ans:
        ans.writelines(answers)


# ../models/classification/TOEFL_iter_4000.pt
def load_model(filename):
    rnn_clf = RNNSequenceClassifier(num_classes=2, embedding_dim=300 + 1024 + 50, hidden_size=300, num_layers=1,
                                    bidir=True,
                                    dropout1=0.3, dropout2=0.2, dropout3=0.2)
    rnn_clf.load_state_dict(torch.load(filename))
    rnn_clf.cuda()
    return rnn_clf


def pred_and_write(filename):
    clf = load_model(filename)
    preds = predict_toefl(clf)
    write_predictions_to_answer_file(preds)
