import numpy as np
import random
import read_data as data_import
import model
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

corpus = "train-test.txt"

N = 3
word_list,w2i,i2w = data_import.read_text(corpus, N)
dim = 50
input_size = (N - 1) * dim
hidden_size = dim * 10
num_classes = len(w2i)
print ('There are',num_classes,'classes')
num_epochs = 30

ngram = data_import.generate_context(N, word_list) # Create ngrams
print ('Created ngrams!')

mlp = model.Net(dim, input_size, hidden_size, num_classes)
trained_model = model.train(N,num_epochs,ngram,w2i,mlp)

start_sentence = [i2w[w2i['<s>']] for i in range(N-1)]
sentence = model.next_word(N,start_sentence,w2i,i2w,trained_model)

while sentence[-1] != '</s>':
    sentence = model.next_word(N,sentence,w2i,i2w,trained_model)

print (sentence)
