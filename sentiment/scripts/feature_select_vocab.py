import numpy as np
import sys
import math
from create_dictionary import create_dictionary

BOW = "out_bag_of_words_5.csv"
DICTIONARY = "out_vocab_5.txt"
CLASSES = "out_classes_5.txt"
N = 50

def get_distributions(sentences, classes, len_vocab):
    dist_pos = [len_vocab]
    word_count_pos = 0
    dist_neg = [len_vocab]
    word_count_neg = 0

    p_pos = float(sum(classes)) / float(len(classes))

    for i in range(len(sentences)):
        sentence = sentences[i]
        word_count = sum(sentence)
        if classes[i] == 0:
            dist_neg += sentence
            word_count_neg += word_count
        else:
            dist_pos += sentence
            word_count_pos += word_count

    dist_neg = [e / float(word_count_neg) for e in dist_neg]
    dist_pos = [e / float(word_count_pos) for e in dist_pos]

    return (dist_neg, dist_pos, p_pos)

def main():
    word_to_id, vocab = create_dictionary(DICTIONARY)
    sentences = np.loadtxt(BOW, dtype=float, delimiter=",")
    classes = np.loadtxt(CLASSES, dtype=int)

    new_vocab_length = int(sys.argv[1])
    dist_neg, dist_pos, p = get_distributions(sentences, classes, len(vocab))

    pmis_pos = []
    pmis_neg = []
    print len(vocab), len(pmis_pos), len(dist_neg), len(dist_pos)
    for i in range(len(vocab)):
        # this below assumes p(positive) = .5
        pmis_pos.append((math.log(dist_pos[i] / (dist_neg[i] + dist_pos[i])), i))
        pmis_neg.append((math.log(dist_neg[i] / (dist_neg[i] + dist_pos[i])), i))

    pmis_pos.sort()
    pmis_neg.sort()

    new_vocab = []
    for i in range(new_vocab_length / 2):
        _, indp = pmis_pos[i]
        _, indn = pmis_neg[i]
        new_vocab.append(vocab[indn])
        new_vocab.append(vocab[indp])

    np.savetxt("fs_vocab.txt", new_vocab, fmt="%s")

main()