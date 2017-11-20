import numpy as np
from collections import defaultdict

def read_text(fname, N, max_lines=np.inf):
    """
    Reads in the data in fname and returns it as
    one long list of words.
    """
    data = []
    i2w = dict()
    w2i = dict()

    with open(fname, "r") as fh:
        for k, line in enumerate(fh):
            if k > max_lines:
                break
            words = line.strip().split()

            for sentence_start in range(N-1):
                data.append('<s>')

            for word in words:
                data.append(word.lower())

            data.append('</s>')

    vocab = set(data)

    for i, word in enumerate(vocab):
        w2i[word] = i
        i2w[i] = word

    if len(w2i) != len(i2w):
        raise NotImplementedError('Length of w2i and i2w are not the same!!!')

    return data, w2i, i2w

def get_sentence_list(data):

    sentence_list = []
    temp_sentence = []
    for i in data:
        temp_sentence.append(i)
        if i == '</s>':
            sentence_list.append(temp_sentence)
            temp_sentence = []

    return sentence_list

def read_and_create_dictionaries(fname, vector_file, max_lines=np.inf):
    """
    Reads in the data in fname and returns it as
    one long list of words.
    """
    data = []
    i2w = dict()
    w2i = defaultdict(lambda : len(w2i))
    #w2i = dict()

    with open(fname, "r") as fh:
        for k, line in enumerate(fh):
            if k > max_lines:
                break
            words = line.strip().split()

            for word in words:
                data.append(word.lower())
                i2w[w2i[word]] = word

    # For now, we manually add the word '<unk>'
    i2w[w2i['<unk>']] = '<unk>'

    w2v = defaultdict(lambda: len(w2i))

    with open (vector_file, "r") as fh:
        for k, line in enumerate(fh):
            words = line.strip().split()

            empty_vector = []

            for i in range (1,len(words)):
                empty_vector.append(float(words[i]))

            w2v[words[0]] = np.asarray(empty_vector)

    w2v['<unk>'] = np.random.rand(50,) # Add <unk> as a random vector

    if len(w2i) != len(i2w):
        raise NotImplementedError('Length of w2i and i2w are not the same!!!')

    return data,w2i,i2w, w2v

def generate_context(N, word_list):
    if N == 2:
        bigram = [([word_list[i]], word_list[i + 1]) for i in range(len(word_list) - 1)]
        return bigram
    elif N == 3:
        trigram = [([word_list[i], word_list[i + 1]], word_list[i + 2]) for i in range(len(word_list) - 2)]
        return trigram
    elif N == 4:
        quadgram = [([word_list[i], word_list[i + 1], word_list[i + 2]], word_list[i + 3]) for i in
                    range(len(word_list) - 3)]
        return quadgram
    elif N == 5:
        fivegram = [([word_list[i], word_list[i + 1], word_list[i + 2], word_list[i + 3]], word_list[i+4]) for i in
                    range(len(word_list) - 4)]
        return fivegram
    else:
        raise NotImplementedError('N should be 2, 3, 4 or 5')

'''
corpus = "train-test.txt"
vector_file = "glove.6B.50d.txt"

data,w2i,i2w,w2v = read_and_create_dictionaries(corpus,vector_file)
print ('done')

for word in data:
    print (len(w2v[word]))
'''