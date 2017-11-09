import numpy as np
from collections import defaultdict

def read_text(fname, max_lines=np.inf):
    """
        Reads in the data in fname and returns it as
        one long list of words.
        """
    data = []
    i2w = dict()
    w2i = defaultdict(lambda : len(w2i))

    with open(fname, "r") as fh:
        for k, line in enumerate(fh):
            if k > max_lines:
                break
            words = line.strip().split()

            for word in words:
                data.append(word.lower())
                i2w[w2i[word]] = word

    return data,w2i,i2w

def write_vectors(vectors,raw_corpus,outname, max_lines=np.inf):

    word_list,_,_ = read_text(raw_corpus)

    necessary_vectors = []

    with open(vectors, "r") as fh:
        for k , line in enumerate(fh):
            if k > max_lines:
                break
            words = line.strip().split()

            for word in word_list:
                if word == words[0]:
                    if words not in necessary_vectors:
                        necessary_vectors.append(words)

    write_file = open(outname,"w")

    for word in necessary_vectors:
        write_file.write(" ".join(word))
        write_file.write('\n')
    write_file.close()

    return

def create_dictionaries(vector_file):
    w2v = defaultdict(lambda: len(w2i))
    v2w = dict()

    with open (vector_file, "r") as fh:
        for k, line in enumerate(fh):
            words = line.strip().split()

            empty_vector = []

            for i in range (1,len(words)):
                empty_vector.append(float(words[i]))

            w2v[words[0]] = np.asarray(empty_vector)

    word_list, w2i, i2w = read_text(raw_corpus)
    return w2v,w2i,i2w,word_list

raw_corpus = "ted-train-test.txt"
vectors = "glove.6B.50d.txt"
outname = "necessary-vectors.txt"
write_vectors(vectors,raw_corpus,outname, max_lines=np.inf)
w2v,w2i,i2w,vocab_size = create_dictionaries(outname)



