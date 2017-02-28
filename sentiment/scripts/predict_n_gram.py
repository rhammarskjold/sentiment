import numpy as np
from create_dictionary import create_dictionary
from preprocessSentences import tokenize_corpus 
from preprocessSentences import find_wordcounts
import sys
import math
import random
import matplotlib.pyplot as plt
from plot_roc import plot_roc

SMOOTHING = 1.0
BOW = "out_bag_of_words_5.csv"
DICTIONARY = "fs_vocab.txt"
CLASSES = "out_classes_5.txt"
TRAINING = "./train.txt"
TESTS = "./test.txt"
EXPERIMENT_STEP = 20


""" takes in the size of the vocabulary and reads in the sentences to create
    the bigram distributions and the p that any given sentence is positive"""
def get_distributions(docs, classes, len_vocab, n):
    dist_pos = []
    word_counts_pos = []
    dist_neg = []
    word_counts_neg = len_vocab * SMOOTHING

    for i in range(len_vocab + 2):
        a = []
        b = []
        for j in range(len_vocab + 2):
            a.append(SMOOTHING)
            b.append(SMOOTHING)
        dist_neg.append(a)
        dist_pos.append(b)

    pos_count = 0
    r = list(range(len(docs)))
    random.shuffle(r)
    for i in range(n):
        doc = docs[r[i]]
        last_word = len_vocab + 1
        for word in doc:
            if int(classes[r[i]]) == 0:
                dist_neg[last_word][word] += 1
            else:
                dist_pos[last_word][word] += 1
            last_word = word
        if int(classes[r[i]]) != 0:
            pos_count += 1

    dist_neg_logs = [[math.log(e / float(sum(d))) for e in d] for d in dist_neg]
    dist_pos_logs = [[math.log(e / float(sum(d))) for e in d] for d in dist_pos]
    # print pos_count, n
    p_pos = float(pos_count) / float(n)
    return (dist_neg_logs, dist_pos_logs, p_pos)

def get_data(path, vocab, word_to_id):
    docs, classes, _, _ = tokenize_corpus(path, train=True)
    docsid = []
    for doc in docs:
        ids = []
        for word in doc:
            if word in word_to_id.keys():
                ids.append(word_to_id[word])
            else:
                ids.append(len(vocab))
        docsid.append(ids)
    return (docsid, classes)

def find_p(observed, dist):
    p = 0.0
    last_word = len(dist) - 1
    for word in observed:
        p += dist[last_word][word]
        last_word = word
    return p

def predict(test_case, dist_pos, dist_neg, p_pos):
    p_given_neg = find_p(test_case, dist_neg)
    p_given_pos = find_p(test_case, dist_pos)
    # print p_given_neg, p_given_pos
    return -math.log(1.0-p_pos) - p_given_neg - math.log(p_pos) + p_given_pos

def main():
    # create word dictionaries
    word_to_id, vocab = create_dictionary(DICTIONARY)

    # training data
    train_docs, train_classes = get_data(TRAINING, vocab, word_to_id)

    # find test data
    test_docs, test_classes = get_data(TESTS, vocab, word_to_id)

    # train
    dist_neg, dist_pos, p_pos = get_distributions(train_docs, train_classes, len(vocab), 2400)

    #print dist_pos[55]
    # test
    scores = []
    for i in range(len(test_classes)):
        test_case = test_docs[i]
        scores.append(predict(test_case, dist_pos, dist_neg, p_pos))

    # print "%d out of %d tests correct predicted %d neg" % (correct, len(test_classes), zeroes)

    plot_roc(scores, [int(cl) for cl in test_classes], "bigram")
    np.savetxt("n_gram_scores.txt", scores)
main()