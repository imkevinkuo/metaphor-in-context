import csv
import os
import nltk
import jsonlines
from tqdm import tqdm

VUA_TRAIN_POS_DIR = "../corpora/VUA_corpus/naacl_flp_skll_train_datasets/all_pos"
VUA_TEST_POS_DIR = "../corpora/VUA_corpus/naacl_flp_skll_test_datasets/all_pos"
VUA_TRAIN_SENTENCES = "../corpora/VUA_corpus/vuamc_corpus_train.csv"
VUA_TEST_SENTENCES = "../corpora/VUA_corpus/vuamc_corpus_test.csv"
VUA_PTOKS_TRAIN = "../corpora/VUA_corpus/all_pos_tokens.csv"
VUA_PTOKS_TEST = "../corpora/VUA_corpus/all_pos_tokens_test.csv"
TOEFL_TRAIN = "../corpora/toefl_sharedtask_dataset"
TOEFL_TEST = "../corpora/toefl_sharedtask_evaluation_kit"


def load_raw_train_toefl():
    pos_and_label = {}
    with jsonlines.open(f'{TOEFL_TRAIN}/toefl_skll_train_features/all_pos/P.jsonlines') as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            pos_tag = nltk.tag.map_tag('en-ptb', 'universal', obj['x']['stanford_postag'])
            label = obj['y']
            pos_and_label[(txt_id, sent_id, int(word_id))] = (pos_tag, label)

    raw_train_toefl = []
    sentences = load_toefl_sents(TOEFL_TRAIN)
    for txt_id, sent_id in tqdm(sentences):
        # Get POS tags and labels for each word
        sent_txt = sentences[(txt_id, sent_id)].replace('M_', '')
        if sent_txt != '':
            sent_tok = sent_txt.split()
            pos_tags = [nltk.tag.map_tag('en-ptb', 'universal', tag) for word, tag in nltk.pos_tag(sent_tok)]
            labels = [0 for tok in sent_tok]
            for i in range(len(sent_tok)):
                word_id = i + 1
                txt_sent_word_id = (txt_id, sent_id, word_id)
                if txt_sent_word_id in pos_and_label:
                    pos_tags[i], labels[i] = pos_and_label[txt_sent_word_id]
            raw_train_toefl.append([sent_txt, labels, pos_tags])
    return raw_train_toefl


def load_raw_test_toefl():
    pos_dict = {}
    with jsonlines.open(f'{TOEFL_TEST}/toefl_skll_test_features_no_labels/all_pos/P.jsonlines') as reader:
        for obj in reader:
            txt_id, sent_id, word_id, word = obj['id'].split("_")
            pos_tag = nltk.tag.map_tag('en-ptb', 'universal', obj['x']['stanford_postag'])
            pos_dict[(txt_id, sent_id, int(word_id))] = pos_tag

    raw_test_toefl = []
    sentences = load_toefl_sents(TOEFL_TEST)
    for txt_id, sent_id in tqdm(sentences):
        # Get POS tags and labels for each word
        sent_txt = sentences[(txt_id, sent_id)].replace('M_', '')
        if sent_txt != '':
            sent_tok = sent_txt.split()
            pos_tags = [nltk.tag.map_tag('en-ptb', 'universal', tag) for word, tag in nltk.pos_tag(sent_tok)]
            for i in range(len(sent_tok)):
                word_id = i + 1
                txt_sent_word_id = (txt_id, sent_id, word_id)
                if txt_sent_word_id in pos_dict:
                    pos_dict[i] = pos_dict[txt_sent_word_id]
            raw_test_toefl.append([sent_txt, pos_tags, (txt_id, sent_id)])
    return raw_test_toefl


def load_toefl_sents(directory):
    sentences = {}
    for filename in os.listdir(f'{directory}/essays'):
        fileid = filename.split('.')[0]
        with open(f'{directory}/essays/{filename}') as f:
            lines = [line.rstrip() for line in f]
            for i in range(len(lines)):
                sentences[(fileid, i + 1)] = lines[i]
    return sentences


def load_toefl_ptoks():
    vtoks = []
    with open(f"{TOEFL_TEST}/toefl_all_pos_test_tokens.csv") as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            txt_id, sent_id, verb_id, verb = line.split("_")
            vtoks.append("_".join((txt_id, sent_id, verb_id)))
    return vtoks


# https://github.com/EducationalTestingService/metaphor/tree/master/VUA-shared-task
# Download the SKLL baseline features (both train/test) and put them in metaphor-in-context/corpora/VUA_corpus.
# These contain POS tags which are needed for the SEQ model to work.
def load_raw_train_vua():
    pos_and_label = {}
    for genre_dir in os.listdir(VUA_TRAIN_POS_DIR):
        with jsonlines.open(f'{VUA_TRAIN_POS_DIR}/{genre_dir}/train/P.jsonlines') as reader:
            for obj in reader:
                txt_id, sent_id, word_id = obj['id'].split("_")
                pos_tag = obj['x']['postag']
                label = obj['y']
                pos_and_label[(txt_id, sent_id, int(word_id))] = (pos_tag, label)

    raw_train_vua = []
    sentences = load_vua_sents(VUA_TRAIN_SENTENCES)
    for txt_id, sent_id in tqdm(sentences):
        # Get POS tags and labels for each word
        sent_txt = sentences[(txt_id, sent_id)].replace('M_', '')
        if sent_txt != '':
            sent_tok = sent_txt.split()
            pos_tags = [nltk.tag.map_tag('en-ptb', 'universal', tag) for word, tag in nltk.pos_tag(sent_tok)]
            labels = [0 for tok in sent_tok]
            for i in range(len(sent_tok)):
                word_id = i+1
                txt_sent_word_id = (txt_id, sent_id, word_id)
                if txt_sent_word_id in pos_and_label:
                    pos_tags[i], labels[i] = pos_and_label[txt_sent_word_id]
            raw_train_vua.append([sent_txt, labels, pos_tags])
    return raw_train_vua


def load_raw_test_vua():
    pos_dict = {}
    for genre_dir in os.listdir(VUA_TEST_POS_DIR):
        with jsonlines.open(f'{VUA_TEST_POS_DIR}/{genre_dir}/test/P.jsonlines') as reader:
            for obj in reader:
                txt_id, sent_id, word_id = obj['id'].split("_")
                pos_tag = obj['x']['postag']
                pos_dict[(txt_id, sent_id, int(word_id))] = pos_tag

    raw_test_vua = []
    sentences = load_vua_sents(VUA_TEST_SENTENCES)
    for txt_id, sent_id in tqdm(sentences):
        # Get POS tags and labels for each word
        sent_txt = sentences[(txt_id, sent_id)].replace('M_', '')
        if sent_txt != '':
            sent_tok = sent_txt.split()
            pos_tags = [nltk.tag.map_tag('en-ptb', 'universal', tag) for word, tag in nltk.pos_tag(sent_tok)]
            for i in range(len(sent_tok)):
                word_id = i + 1
                txt_sent_word_id = (txt_id, sent_id, word_id)
                if txt_sent_word_id in pos_dict:
                    pos_dict[i] = pos_dict[txt_sent_word_id]
            raw_test_vua.append([sent_txt, pos_tags, (txt_id, sent_id)])
    return raw_test_vua


def get_pos_set(raw_train, raw_test):
    pos_set = set()
    for example in raw_train:
        pos_tags = example[-1]
        pos_set.update(pos_tags)

    for example in raw_test:
        pos_tags = example[-1]
        pos_set.update(pos_tags)
    return pos_set


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


# ../corpora/VUA_corpus/all_pos_tokens.csv
# ../corpora/VUA_corpus/all_pos_tokens_test.csv
def load_vua_ptoks(filename):
    """ Return a list of txt_sent_ids, and if on training set include label."""
    examples = []
    with open(filename, encoding='utf-8') as f:
        lines = csv.reader(f)
        for line in lines:
            examples.append(line)
    return examples
