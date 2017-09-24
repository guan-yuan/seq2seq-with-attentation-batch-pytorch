# coding: utf-8
import torch
import copy
import pickle
USE_CUDA = True

PAD_token = 0
SOS_token = 1
EOS_token = 2
FILE_POST = "data/weibo_post_zh.txt"
FILE_RESPONSE = "data/weibo_response_zh.txt"

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


def read_langs(voc_name):
    print("Reading lines...")

    with open(FILE_POST) as f1:
        post = f1.readlines()
    print("len of post: ", len(post))

    with open(FILE_RESPONSE) as f2:
        response = f2.readlines()
    print("len of response: ", len(response))

    pairs = [[i.strip("\n"), j.strip("\n")] for i, j in zip(post, response)]
    voc = Voc(voc_name)

    return voc, pairs


MIN_LENGTH = 3
MAX_LENGTH = 20


def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH and len(pair[1]) >= MIN_LENGTH and len(
                pair[1]) <= MAX_LENGTH:
            filtered_pairs.append(pair)
    return filtered_pairs


def prepare_data(voc_name):
    voc, pairs = read_langs(voc_name)
    print("Read %d sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        voc.index_words(pair[0])
        voc.index_words(pair[1])

    print('Indexed %d words!' % (voc.n_words))
    return voc, pairs


voc, pairs = prepare_data('weibo')
voc_trimed = copy.deepcopy(voc)

MIN_COUNT = 120

voc_trimed.trim(MIN_COUNT)


keep_pairs = []

for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True

    for word in input_sentence.split(' '):
        if word not in voc_trimed.word2index:
            keep_input = False
            break

    for word in output_sentence.split(' '):
        if word not in voc_trimed.word2index:
            keep_output = False
            break

    # Remove if pair doesn't match input and output conditions
    if keep_input and keep_output:
        keep_pairs.append(pair)

print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
pairs = keep_pairs
with open("save/pairs.pickle", "wb") as f:
    pickle.dump(pairs, f)

with open("save/voc_trimed.pickle", "wb") as f:
    pickle.dump(voc_trimed, f)

with open("save/voc.pickle", "wb") as f:
    pickle.dump(voc, f)

with open("save/pairs.txt", "w") as f:
    count = 0
    for i in pairs:
        f.write(i[0] + "\n")
        f.write(i[1] + "\n")
        f.write("\n")
        count += 1
    print("num of pairs:", count)