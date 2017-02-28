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
DICTIONARY = "out_vocab_5.txt"
CLASSES = "out_classes_5.txt"
TESTS = "test.txt"
EXPERIMENT_STEP = 10

def to_binom_verctors(sentences):
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if word > 0:
                sentences[i][j] = 1

""" takes in the size of the vocabulary and reads in the sentences to create
    the multinomial distributions and the p that any given sentence is positive"""
def get_distributions(sentences, classes, len_vocab, n):
    dist_pos = list()
    word_count_pos = len_vocab * SMOOTHING
    dist_neg = list()
    word_count_neg = len_vocab * SMOOTHING

    for i in range(len_vocab):
        dist_neg.append(SMOOTHING)
        dist_pos.append(SMOOTHING)

    p_pos = float(sum(classes)) / float(len(classes))
    r = list(range(len(sentences)))
    random.shuffle(r)
    #print r
    if n == -1:
        n = len(sentences)
    for i in range(n):
        sentence = sentences[r[i]]
        word_count = sum(sentence)
        if classes[r[i]] == 0:
            dist_neg += sentence
            word_count_neg += word_count
        else:
            dist_pos += sentence
            word_count_pos += word_count

    dist_neg = [math.log(e / float(n)) for e in dist_neg]
    dist_pos = [math.log(e / float(n)) for e in dist_pos]

    return (dist_neg, dist_pos, p_pos)

def get_tests(vocab):
    docs, classes, _ = tokenize_corpus("./test.txt", train=False)
    bow = find_wordcounts(docs, vocab)
    return (bow, classes)

def find_p(observed, dist):
    p = 0.0
    for i, x in enumerate(observed):
        if x > 0:
            p += dist[i]
    return p

def predict(test_case, dist_pos, dist_neg, p_pos):
    p_given_neg = find_p(test_case, dist_neg)
    p_given_pos = find_p(test_case, dist_pos)
    return  - math.log(1.0-p_pos) - p_given_neg + (math.log(p_pos) + p_given_pos)


def main():
    # create word dictionaries
    word_to_id, vocab = create_dictionary(DICTIONARY)

    # training data
    sentences = np.loadtxt(BOW, dtype=float, delimiter=",")
    to_binom_verctors(sentences)
    classes = np.loadtxt(CLASSES, dtype=int)

    # find test data
    test_bow, test_classes = get_tests(vocab)


    dist_neg, dist_pos, p_pos = get_distributions(sentences, classes, len(vocab), 2400)

    # test
    scores = []
    for i in range(len(test_classes)):
        test_case = test_bow[i]
        scores.append(predict(test_case, dist_pos, dist_neg, p_pos))

    # print "%d out of %d tests correct predicted %d neg" % (correct, len(test_classes), zeroes)
    plot_roc(scores, [int(cl) for cl in test_classes], "binomial naive bayes")
    np.savetxt("binomial_scores.txt", scores)

main()