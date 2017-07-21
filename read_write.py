import sys
import gzip
import numpy
import math

from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename, top_vocab_only = 1e6):
    word_vecs = {}
    if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
    else: file_object = open(filename, 'r')

    for line_num, line in enumerate(file_object):
        if line_num > top_vocab_only and top_vocab_only > 0:
            break
        line = line.strip().lower()
        word = line.split()[0]
        word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
        for index, vec_val in enumerate(line.split()[1:]):
            word_vecs[word][index] = float(vec_val)
        word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)

    sys.stderr.write("Vectors read from: "+filename+" \n")
    return word_vecs
