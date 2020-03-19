import random
from tqdm import tqdm

from util import get_num_lines, get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, write_predictions, get_performance_VUAverb_val, \
    get_performance_VUAverb_test, get_performance_VUA_test
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate, predict
from model import RNNSequenceModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import csv
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
raw_train_vua = data_parser.load_raw_train_vua()
raw_test_vua = data_parser.load_raw_test_vua()
pos_set = data_parser.get_pos_set(raw_train_vua, raw_test_vua)

# embed the pos tags
pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)

for i in range(len(raw_train_vua)):
    raw_train_vua[i][2] = index_sequence(pos2idx, raw_train_vua[i][2])

for i in range(len(raw_test_vua)):
    raw_test_vua[i][1] = index_sequence(pos2idx, raw_test_vua[i][1])

print('size of training, test set: ', len(raw_train_vua), len(raw_test_vua))

"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_train_vua + raw_test_vua)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
elmos_train_vua = h5py.File('../elmo/VUA_train2.hdf5', 'r')
elmos_test_vua = h5py.File('../elmo/VUA_test2.hdf5', 'r')
# no suffix embeddings for sequence labeling
suffix_embeddings = None

'''
2. 2
embed the datasets
'''
# random.seed(0)
# random.shuffle(raw_train_vua)

sentence_to_index_train = ast.literal_eval(elmos_train_vua['sentence_to_index'][0])
labels = [example[1] for example in raw_train_vua]
poss = [example[2] for example in raw_train_vua]
sentences = [embed_indexed_sequence(example[0], example[2], word2idx, glove_embeddings,
                                    elmos_train_vua, suffix_embeddings, sentence_to_index_train[example[0]])
             for example in raw_train_vua]


def train_model(train_dataloader_vua, val_dataloader_vua, fold_num):
    optimal_f1s = []
    optimal_ps = []
    optimal_rs = []
    optimal_accs = []
    predictions_all = []

    RNNseq_model = RNNSequenceModel(num_classes=2, embedding_dim=300 + 1024, hidden_size=300, num_layers=1, bidir=True,
                                    dropout1=0.5, dropout2=0, dropout3=0.1)
    # Move the model to the GPU if available
    if using_GPU:
        RNNseq_model = RNNseq_model.cuda()
    # Set up criterion for calculating loss
    loss_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    rnn_optimizer = optim.Adam(RNNseq_model.parameters(), lr=0.005)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 10

    '''
    3. 2
    train model
    '''
    train_loss = []
    val_loss = []
    performance_matrix = None
    val_f1s = []
    train_f1s = []
    # A counter for the number of gradient updates
    num_iter = 0
    comparable = []
    for epoch in tqdm(range(num_epochs)):
        # print("Starting epoch {}".format(epoch + 1))
        for (_, example_text, example_lengths, example_labels) in train_dataloader_vua:
            example_text = Variable(example_text)
            example_lengths = Variable(example_lengths)
            example_labels = Variable(example_labels)
            if using_GPU:
                example_text = example_text.cuda()
                example_lengths = example_lengths.cuda()
                example_labels = example_labels.cuda()
            # predicted shape: (batch_size, seq_len, 2)
            predicted = RNNseq_model(example_text, example_lengths)
            batch_loss = loss_criterion(predicted.view(-1, 2), example_labels.view(-1))
            rnn_optimizer.zero_grad()
            batch_loss.backward()
            rnn_optimizer.step()
            num_iter += 1
            # Calculate validation and training set loss and accuracy every 200 gradient updates
            if num_iter % 200 == 0:
                avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader_vua, RNNseq_model,
                                                             loss_criterion, using_GPU)
                val_loss.append(avg_eval_loss)
                val_f1s.append(performance_matrix[:, 2])
                print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))
                filename = f"../models/sequence/VUA_fold_{fold_num}_iter_{num_iter}.pt"
                torch.save(RNNseq_model.state_dict(), filename)
                # avg_eval_loss, performance_matrix = evaluate(idx2pos, train_dataloader_vua, RNNseq_model,
                #                                              loss_criterion, using_GPU)
    #             train_loss.append(avg_eval_loss)
    #             train_f1s.append(performance_matrix[:, 2])
    #             print("Iteration {}. Training Loss {}.".format(num_iter, avg_eval_loss))

    """
    for additional training
    """
    rnn_optimizer = optim.Adam(RNNseq_model.parameters(), lr=0.0001)
    for epoch in range(10):
        # print("Starting epoch {}".format(epoch + 1))
        for (__, example_text, example_lengths, example_labels) in train_dataloader_vua:
            example_text = Variable(example_text)
            example_lengths = Variable(example_lengths)
            example_labels = Variable(example_labels)
            if using_GPU:
                example_text = example_text.cuda()
                example_lengths = example_lengths.cuda()
                example_labels = example_labels.cuda()
            # predicted shape: (batch_size, seq_len, 2)
            predicted = RNNseq_model(example_text, example_lengths)
            batch_loss = loss_criterion(predicted.view(-1, 2), example_labels.view(-1))
            rnn_optimizer.zero_grad()
            batch_loss.backward()
            rnn_optimizer.step()
            num_iter += 1
            # Calculate validation and training set loss and accuracy every 200 gradient updates
            if num_iter % 200 == 0:
                avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader_vua, RNNseq_model,
                                                             loss_criterion, using_GPU)
                val_loss.append(avg_eval_loss)
                # val_f1s.append(performance_matrix[:, 2])
                print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))
                filename = f"../models/sequence/VUA_fold_{fold_num}_iter_{num_iter}.pt"
                torch.save(RNNseq_model.state_dict(), filename)

    return RNNseq_model, loss_criterion


def train_k_fold(k):
    clfs = []
    fold_size = int(len(raw_train_vua) / k)

    for i in range(k):
        val_indices = [z for z in range(i * fold_size, (i + 1) * fold_size)]
        embedded_train_vua = [[sentences[j], poss[j], labels[j]] for j in range(len(sentences)) if
                              j not in val_indices]
        embedded_val_vua = [[sentences[j], poss[j], labels[j]] for j in range(len(sentences)) if
                            j in val_indices]

        '''
        2. 3
        set up Dataloader for batching
        '''
        # Separate the input (embedded_sequence) and labels in the indexed train sets.
        # raw_train_vua: sentence, label_seq, pos_seq
        # embedded_train_vua: embedded_sentence, pos, labels
        train_dataset_vua = TextDataset([example[0] for example in embedded_train_vua],
                                        [example[1] for example in embedded_train_vua],
                                        [example[2] for example in embedded_train_vua])
        val_dataset_vua = TextDataset([example[0] for example in embedded_val_vua],
                                      [example[1] for example in embedded_val_vua],
                                      [example[2] for example in embedded_val_vua])

        # Data-related hyperparameters
        batch_size = 64
        # Set up a DataLoader for the training and validation sets
        train_dataloader_vua = DataLoader(dataset=train_dataset_vua, batch_size=batch_size, shuffle=True,
                                          collate_fn=TextDataset.collate_fn)
        val_dataloader_vua = DataLoader(dataset=val_dataset_vua, batch_size=batch_size,
                                        collate_fn=TextDataset.collate_fn)
        clf, crit = train_model(train_dataloader_vua, val_dataloader_vua, i)
        clfs.append(clf)
    return clfs


# Test set
sentence_to_index_test = ast.literal_eval(elmos_test_vua['sentence_to_index'][0])
embedded_test_vua = [[embed_indexed_sequence(example[0], example[1], word2idx, glove_embeddings,
                                             elmos_test_vua, suffix_embeddings, sentence_to_index_test[example[0]]),
                      example[1], example[2]] for example in raw_test_vua]


def test_model(RNNseq_model, loss_criterion, val_dataloader_vua):
    avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader_vua, RNNseq_model, loss_criterion, using_GPU)
    return performance_matrix, pos2idx


def predict_vua_allpos(RNNseq_model):
    preds = {}
    for (embed, pos_seq, txt_sent_id) in embedded_test_vua:
        ex_data = TextDataset([embed], [pos_seq], [[0 for pos in pos_seq]])
        ex_dataloader = DataLoader(dataset=ex_data, batch_size=1, collate_fn=TextDataset.collate_fn)
        pred = predict(ex_dataloader, RNNseq_model, using_GPU)
        preds[txt_sent_id] = pred[0][0]
    return preds


def write_preds_to_answers(preds):
    ptoks = data_parser.load_vua_ptoks(data_parser.VUA_PTOKS_TEST)
    answers = []
    for ptok in ptoks:
        txt_id, sent_id, word_id = ptok[0].split("_")
        prediction = preds[(txt_id, sent_id)][int(word_id)-1]
        answers.append(f"{ptok[0]},{prediction}\n")

    with open("answer.txt", "w") as ans:
        ans.writelines(answers)


def load_model(filename):
    RNNseq_model = RNNSequenceModel(num_classes=2, embedding_dim=300 + 1024, hidden_size=300, num_layers=1, bidir=True,
                                    dropout1=0.5, dropout2=0, dropout3=0.1)
    RNNseq_model.load_state_dict(torch.load(filename))
    if using_GPU:
        RNNseq_model.cuda()
    return RNNseq_model
