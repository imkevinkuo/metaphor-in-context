import csv
import os
import nltk
TOEFL_TRAIN = "../corpora/toefl_sharedtask_dataset/essays"
TOEFL_TEST = "../corpora/toefl_sharedtask_evaluation_kit"
VUA_VERB_TOKS_TRAIN = "../corpora/VUA_corpus/verb_tokens.csv"
VUA_SENTENCES_TRAIN = "../corpora/VUA_corpus/vuamc_corpus_train.csv"
VUA_VERB_TOKS_TEST = "../corpora/VUA_corpus/verb_tokens_test.csv"
VUA_SENTENCES_TEST = "../corpora/VUA_corpus/vuamc_corpus_test.csv"


# We may need to run some kind of spellcheck / autocorrect on this text...
def load_raw_toefl_train():
    raw_toefl_train = []
    example_ids = []
    for filename in os.listdir(TOEFL_TRAIN):
        fileid = filename.split('.')[0]
        with open(f"{TOEFL_TRAIN}/{filename}") as f:
            lines = [line.rstrip() for line in f]
            for i in range(len(lines)):
                tok_sent = lines[i].split()
                # TODO: need to edit this because M_ screws it up
                pos_tags = [(word, nltk.tag.map_tag('en-ptb', 'universal', pos_tag)) for word, pos_tag in nltk.pos_tag(tok_sent)]
                for j in range(len(tok_sent)):
                    if pos_tags[j][1] == 'VERB':
                        label = 0
                        if tok_sent[j].startswith("M_"):
                            label = 1
                        raw_toefl_train.append([lines[i], j, label])  # sentence, verb_idx, label
                        example_ids.append("_".join((fileid, str(i+1), str(j+1))))
    return raw_toefl_train, example_ids


def load_raw_toefl_test():
    tok_mapping = {}
    example_ids = []
    for filename in os.listdir(f"{TOEFL_TEST}/essays"):
        fileid = filename.split('.')[0]
        with open(f"{TOEFL_TEST}/essays/{filename}") as f:
            lines = [line.rstrip() for line in f]
            for i in range(len(lines)):
                tok_mapping[(fileid, str(i+1))] = lines[i]

    raw_toefl_test = []
    with open(f"{TOEFL_TEST}/toefl_verb_test_tokens.csv") as f:
        lines = [line.rstrip() for line in f]
        for line in lines:
            fileid, sent_id, verb_id, verb = line.split("_")
            sentence = tok_mapping[(fileid, sent_id)]
            verb_idx = int(verb_id) - 1
            if verb != sentence.split()[verb_idx]:
                print(f"Mismatch: found {sentence[verb_idx]} instead of {verb}.")
                print(line)
            raw_toefl_test.append([sentence, verb_idx])
            example_ids.append("_".join((fileid, sent_id, verb_id)))
    return raw_toefl_test, example_ids


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
# Then use allennlp to write the ELMo vectors to disk.
# allennlp elmo train_sentences_vua.txt VUA_train.hdf5 --average --cuda-device 0
# allennlp elmo test_sentences_vua.txt VUA_test.hdf5 --average --cuda-device 0
def write_vua_sents_to_txt():
    train_d = load_vua_sents(VUA_SENTENCES_TRAIN)
    vua_train_sents = set()
    for k in train_d:
        sentence = train_d[k].replace('M_', '').rstrip()
        if sentence != '':
            vua_train_sents.add(sentence + '\n')
    with open("train_sentences_vua.txt", 'w') as f:
        f.writelines(vua_train_sents)

    test_d = load_vua_sents(VUA_SENTENCES_TEST)
    vua_test_sents = set()
    for k in test_d:
        sentence = test_d[k].replace('M_', '').rstrip()
        if sentence != '':
            vua_test_sents.add(test_d[k] + '\n')
    with open("test_sentences_vua.txt", 'w') as f:
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
