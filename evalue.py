# coding: utf-8

import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from masked_cross_entropy import *
import train
from train import EncoderRNN
from train import Attn
from train import LuongAttnDecoderRNN

import pickle
from preprocess import Voc
from preprocess import MAX_LENGTH

USE_CUDA = True

PAD_token = 0
SOS_token = 1
EOS_token = 2

with open("save/pairs.pickle", "rb") as f:
    pairs = pickle.load(f)

pairs = pairs[-250:]
targets = [pair[1] for pair in pairs]

fake_pairs1 = []
fake_pairs2 = []
fake_pairs3 = []
fake_pairs4 = []
fake_pairs5 = []
for i in range(len(pairs)):
    post = pairs[i][0]
    targets.remove(pairs[i][1])
    fake_pairs1.append([post, random.choice(targets)])
    fake_pairs2.append([post, random.choice(targets)])
    fake_pairs3.append([post, random.choice(targets)])
    fake_pairs4.append([post, random.choice(targets)])
    fake_pairs5.append([post, random.choice(targets)])
    targets.append(pairs[i][1])

with open("save/MCQ.txt", "w") as f:
    for i in range(len(pairs)):
        f.write("post: " + pairs[i][0] + "\n")
        f.write(pairs[i][1] + "\n")
        f.write(fake_pairs1[i][1] + "\n")
        f.write(fake_pairs2[i][1] + "\n")
        f.write(fake_pairs3[i][1] + "\n")
        f.write(fake_pairs4[i][1] + "\n")
        f.write(fake_pairs5[i][1] + "\n")
        f.write("\n")

with open("save/voc.pickle", "rb") as f:
    voc = pickle.load(f)

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ') if word in voc.word2index] + [EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def iter_pairs(the_iter_pairs, batch_size, count):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = the_iter_pairs[count]
        input_seqs.append(indexes_from_sentence(voc, pair[0]))
        target_seqs.append(indexes_from_sentence(voc, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs) # unzip

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths



def evalue(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, max_length=MAX_LENGTH):
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    return loss.data[0]


# Configure models
attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0
batch_size = 1

n_epochs = len(pairs)
epoch = 0

# Initialize models
encoder = EncoderRNN(voc.n_words, hidden_size, n_layers, dropout=dropout)
decoder = LuongAttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout)
encoder.train(False)
decoder.train(False)

encoder1 = EncoderRNN(voc.n_words, hidden_size, n_layers, dropout=dropout)
decoder1 = LuongAttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout)
encoder1.train(False)
decoder1.train(False)

encoder2 = EncoderRNN(voc.n_words, hidden_size, n_layers, dropout=dropout)
decoder2 = LuongAttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout)
encoder2.train(False)
decoder2.train(False)

encoder3 = EncoderRNN(voc.n_words, hidden_size, n_layers, dropout=dropout)
decoder3 = LuongAttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout)
encoder3.train(False)
decoder3.train(False)

encoder4 = EncoderRNN(voc.n_words, hidden_size, n_layers, dropout=dropout)
decoder4 = LuongAttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout)
encoder4.train(False)
decoder4.train(False)

encoder5 = EncoderRNN(voc.n_words, hidden_size, n_layers, dropout=dropout)
decoder5 = LuongAttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout)
encoder5.train(False)
decoder5.train(False)


# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()
    encoder1.cuda()
    decoder1.cuda()
    encoder2.cuda()
    decoder2.cuda()
    encoder3.cuda()
    decoder3.cuda()
    encoder4.cuda()
    decoder4.cuda()
    encoder5.cuda()
    decoder5.cuda()


import os
if os.path.isfile("save/encoder.pkl"):
    encoder.load_state_dict(torch.load("save/encoder.pkl"))
    encoder1.load_state_dict(torch.load("save/encoder.pkl"))
    encoder2.load_state_dict(torch.load("save/encoder.pkl"))
    encoder3.load_state_dict(torch.load("save/encoder.pkl"))
    encoder4.load_state_dict(torch.load("save/encoder.pkl"))
    encoder5.load_state_dict(torch.load("save/encoder.pkl"))
    print("loaded encoder state_dict!")

if os.path.isfile("save/decoder.pkl"):
    decoder.load_state_dict(torch.load("save/decoder.pkl"))
    decoder1.load_state_dict(torch.load("save/decoder.pkl"))
    decoder2.load_state_dict(torch.load("save/decoder.pkl"))
    decoder3.load_state_dict(torch.load("save/decoder.pkl"))
    decoder4.load_state_dict(torch.load("save/decoder.pkl"))
    decoder5.load_state_dict(torch.load("save/decoder.pkl"))
    print("loaded decoder state_dict!")


NLL_losses = []
while epoch < n_epochs:
    epoch += 1

    #####################################################################################################
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = iter_pairs(pairs, batch_size, epoch-1)

    # Run the train function
    loss0 = evalue(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder)

    #####################################################################################################
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = iter_pairs(fake_pairs1, batch_size, epoch-1)

    # Run the train function
    loss1 = evalue(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder1, decoder1)

    #####################################################################################################
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = iter_pairs(fake_pairs2, batch_size, epoch-1)

    # Run the train function
    loss2 = evalue(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder2, decoder2)

    #####################################################################################################
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = iter_pairs(fake_pairs3, batch_size, epoch-1)

    # Run the train function
    loss3 = evalue(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder3, decoder3)

    #####################################################################################################
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = iter_pairs(fake_pairs4, batch_size, epoch-1)

    # Run the train function
    loss4 = evalue(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder4, decoder4)

    #####################################################################################################
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = iter_pairs(fake_pairs5, batch_size, epoch-1)

    # Run the train function
    loss5 = evalue(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder5, decoder5)

    NLL_losses.append([loss0, loss1, loss2, loss3, loss4, loss5])

print(NLL_losses[:10])
with open("save/NLL_losses.pickle", "wb") as f:
    pickle.dump(NLL_losses, f)

import numpy as np
answers = [0 for i in range(len(pairs))]
predicts = [np.argmin(c) for c in NLL_losses]
bool_results = np.array(answers) == np.array(predicts)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(answers, predicts)
print("accuracy: ", accuracy)

with open("save/MCQ_result.txt", "w") as f:
    f.write("accuracy: %s\n" % accuracy)
    for i in range(len(pairs)):
        f.write("post: " + pairs[i][0] + "\n")
        f.write("%s : %s\n" % (pairs[i][1], NLL_losses[i][0]))
        f.write("%s : %s\n" % (fake_pairs1[i][1], NLL_losses[i][1]))
        f.write("%s : %s\n" % (fake_pairs2[i][1], NLL_losses[i][2]))
        f.write("%s : %s\n" % (fake_pairs3[i][1], NLL_losses[i][3]))
        f.write("%s : %s\n" % (fake_pairs4[i][1], NLL_losses[i][4]))
        f.write("%s : %s\n" % (fake_pairs5[i][1], NLL_losses[i][5]))
        f.write("%s\n" % (bool_results[i]))
        f.write("answer: %s, choice: %s\n" % (answers[i], predicts[i]))
        f.write("\n")


