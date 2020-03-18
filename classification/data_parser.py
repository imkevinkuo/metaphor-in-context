import csv
import os
import nltk
from jsonlines import jsonlines

TOEFL_TRAIN = "../corpora/toefl_sharedtask_dataset"
TOEFL_TEST = "../corpora/toefl_sharedtask_evaluation_kit"
VUA_VERB_TOKS_TRAIN = "../corpora/VUA_corpus/verb_tokens.csv"
VUA_SENTENCES_TRAIN = "../corpora/VUA_corpus/vuamc_corpus_train.csv"
VUA_VERB_TOKS_TEST = "../corpora/VUA_corpus/verb_tokens_test.csv"
VUA_SENTENCES_TEST = "../corpora/VUA_corpus/vuamc_corpus_test.csv"


# We may need to run some kind of spellcheck / autocorrect on this text...
def load_raw_toefl_train():
    pos_and_label = {}
    with jsonlines.open(f'{TOEFL_TRAIN}/toefl_skll_train_features/all_pos/P.jsonlines') as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            pos_tag = obj['x']['stanford_postag']
            label = obj['y']
            pos_and_label[(txt_id, int(sent_id), int(word_id))] = (pos_tag, label)

    toefl_train_sents = {}  # (essay, sentence_id) mapping to sentence
    raw_toefl_train = []

    for filename in os.listdir(f'{TOEFL_TRAIN}/essays'):
        fileid = filename.split('.')[0]
        with open(f'{TOEFL_TRAIN}/essays/{filename}') as f:
            lines = [line.rstrip() for line in f]
            for i in range(len(lines)):
                tok_sent = lines[i].split()
                sentence = lines[i].replace('M_', '')
                toefl_train_sents[(fileid, i + 1)] = sentence
                for j in range(len(tok_sent)):
                    if (fileid, i+1, j+1) in pos_and_label:
                        pos_tag, label = pos_and_label[(fileid, i+1, j+1)]
                        if pos_tag.startswith('V'):
                            raw_toefl_train.append([sentence, j, int(label)])  # sentence, verb_idx, label
    return raw_toefl_train, toefl_train_sents


def load_raw_toefl_test():
    pos_tags = {}
    with jsonlines.open(f'{TOEFL_TEST}/toefl_skll_test_features_no_labels/all_pos/P.jsonlines') as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            pos_tag = obj['x']['stanford_postag']
            pos_tags[(txt_id, int(sent_id), int(word_id))] = pos_tag

    toefl_test_sents = {}  # (essay, sentence_id) mapping to sentence
    raw_toefl_test = []

    for filename in os.listdir(f'{TOEFL_TEST}/essays'):
        fileid = filename.split('.')[0]
        with open(f'{TOEFL_TEST}/essays/{filename}') as f:
            lines = [line.rstrip() for line in f]
            for i in range(len(lines)):
                tok_sent = lines[i].split()
                sentence = lines[i]
                toefl_test_sents[(fileid, i + 1)] = sentence
                for j in range(len(tok_sent)):
                    if (fileid, i+1, j+1) in pos_tags:
                        pos_tag = pos_tags[(fileid, i+1, j+1)]
                        if pos_tag.startswith('V'):
                            raw_toefl_test.append([sentence, j])  # sentence, verb_idx, label
    return raw_toefl_test, toefl_test_sents


# For creating ELMo vectors.
# Run write_toefl_sents_to_txt() in the Python console, which generates two text files.
# Open elmo folder in terminal and use allennlp to write the ELMo vectors to disk.
# allennlp elmo train_sentences_toefl.txt TOEFL_train.hdf5 --average --cuda-device 0
# allennlp elmo test_sentences_toefl.txt TOEFL_test.hdf5 --average --cuda-device 0
def write_toefl_sents_to_txt():
    raw_toefl_train, toefl_train_sents = load_raw_toefl_train()
    raw_toefl_test, toefl_test_sents = load_raw_toefl_test()

    train_sentence_list = []
    for k in toefl_train_sents:
        sentence = toefl_train_sents[k]
        if sentence != '':
            train_sentence_list.append(sentence + '\n')
    with open("../elmo/train_sentences_toefl.txt", 'w') as f:
        f.writelines(train_sentence_list)

    test_sentence_list = []
    for k in toefl_test_sents:
        sentence = toefl_test_sents[k]
        if sentence != '':
            test_sentence_list.append(sentence + '\n')
    with open("../elmo/test_sentences_toefl.txt", 'w') as f:
        f.writelines(test_sentence_list)


# For main_vua at training/test time
def load_raw_train_vua():
    raw_train_vua = []
    verb_toks = load_vua_vtoks(VUA_VERB_TOKS_TRAIN)
    sentences = load_vua_sents(VUA_SENTENCES_TRAIN)
    for vtok in verb_toks:
        txt_sent_verb_id, label = vtok
        txt_id, sent_id, verb_id = txt_sent_verb_id.split("_")
        sent_txt = sentences[(txt_id, sent_id)]
        raw_train_vua.append([sent_txt.replace('M_', ''), int(verb_id)-1, int(label)])
    return raw_train_vua


def load_raw_test_vua():
    raw_test_vua = []
    verb_toks = load_vua_vtoks(VUA_VERB_TOKS_TEST)
    sentences = load_vua_sents(VUA_SENTENCES_TEST)
    for vtok in verb_toks:
        txt_sent_verb_id = vtok[0]
        txt_id, sent_id, verb_id = txt_sent_verb_id.split("_")
        sent_txt = sentences[(txt_id, sent_id)]
        # sent_txt.replace('M_', '')
        raw_test_vua.append([sent_txt, int(verb_id) - 1, txt_sent_verb_id])
    return raw_test_vua


# For creating ELMo vectors.
# Run write_vua_sents_to_txt() in the Python console, which generates two text files.
# Open elmo folder in terminal and use allennlp to write the ELMo vectors to disk.
# allennlp elmo train_sentences_vua.txt VUA_train2.hdf5 --average --cuda-device 0
# allennlp elmo test_sentences_vua.txt VUA_test2.hdf5 --average --cuda-device 0
def write_vua_sents_to_txt():
    train_d = load_vua_sents(VUA_SENTENCES_TRAIN)
    vua_train_sents = set()
    for k in train_d:
        sentence = train_d[k].replace('M_', '').rstrip()
        if sentence != '':
            vua_train_sents.add(sentence + '\n')
    with open("../elmo/train_sentences_vua.txt", 'w') as f:
        f.writelines(vua_train_sents)

    test_d = load_vua_sents(VUA_SENTENCES_TEST)
    vua_test_sents = set()
    for k in test_d:
        sentence = test_d[k].replace('M_', '').rstrip()
        if sentence != '':
            vua_test_sents.add(test_d[k] + '\n')
    with open("../elmo/test_sentences_vua.txt", 'w') as f:
        f.writelines(vua_test_sents)


# ../corpora/VUA_corpus/vuamc_corpus_train.csv
# ../corpora/VUA_corpus/vuamc_corpus_test.csv
def load_vua_sents(filename):
    d = {}
    with open(filename, encoding='utf-8') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            if len(line) > 0:
                d[(line[0], line[1])] = line[2]
    return d


# ../corpora/VUA_corpus/verb_tokens.csv
# ../corpora/VUA_corpus/verb_tokens_test.csv
def load_vua_vtoks(filename):
    """ Return a list of txt_sent_ids, and if on training set include label."""
    examples = []
    with open(filename, encoding='utf-8') as f:
        lines = csv.reader(f)
        for line in lines:
            examples.append(line)
    return examples


# For checking formatted version against the original
def load_vua_eval_sentences():
    d = {}
    with open("../data/VUA/VUA_formatted_test.csv", encoding='utf-8') as f:
        lines = csv.reader(f, delimiter=",")
        next(lines)
        for line in lines:
            if len(line) > 0:
                d[(line[0], line[1])] = line[3]
    return d


def check_formatted_vs_original():
    vua_eval_sentences = load_vua_eval_sentences()
    vua_test_sentences = load_vua_sents(VUA_SENTENCES_TEST)

    for txt_sent_id in vua_eval_sentences:
        sentence = vua_eval_sentences[txt_sent_id]
        if txt_sent_id not in vua_test_sentences:
            print("Not in formatted data:")
            print(txt_sent_id, sentence)
        elif sentence != vua_test_sentences[txt_sent_id]:
            print("Does not match formatted data:")
            print(txt_sent_id, sentence)
