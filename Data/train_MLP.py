import numpy as np
import random
import read_data as data_import
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

corpus = "train.txt"
vector_file = "glove.6B.50d.txt"
word_list,w2i,i2w = data_import.read_text(corpus)

if len(w2i) != len(i2w):
    raise NotImplementedError ('Length of w2i and i2w are not the same!!!')

N = 3
dim = 50
input_size = (N - 1) * dim
hidden_size = dim * 10
num_classes = len(w2i)
print ('There are',num_classes,'classes')
#print ('The wordlist contains',len(word_list),'words')
num_epochs = 10
learning_rate = 0.001

ngram = data_import.generate_context(N, word_list) # Create ngrams
print ('Created ngrams!')

class Net(nn.Module):
    def __init__(self, dim, input_size,hidden_size, num_classes):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(num_classes,dim)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embeds = self.embeddings(x).view((1,-1))
        out = F.tanh(self.l1(embeds))
        out = self.l2(out)
        out = F.softmax(out)
        return out


mlp = Net(dim, input_size, hidden_size, num_classes)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

print ('There are',len(ngram),'ngrams to train')

for epoch in range(num_epochs):

    random.shuffle(ngram)
    total_loss = torch.Tensor([0])

    iter_count = 0
    for context, target in ngram:

        # Define input vector
        input_vector = [w2i[w] for w in context]
        input_vector = autograd.Variable(torch.LongTensor(input_vector))

        if len(input_vector) != (N - 1):
            # Check if input_vector if of the correct dimension
            print ('Context is',context)
            raise NotImplementedError('Length is not N-1, there is probably an unknown word')
        # Done defining input vector

        mlp.zero_grad()
        optimizer.zero_grad()
        output = mlp(input_vector)

        # Calculate the loss and update the weights
        loss = criterion(output, autograd.Variable(torch.LongTensor([w2i[target]])))
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        iter_count += 1

        if iter_count % 100 == 0:
            print ('in epoch',epoch,'it is now at iter',100*iter_count/len(ngram),'%')
    print('After epoch',epoch,'the total loss is',total_loss)