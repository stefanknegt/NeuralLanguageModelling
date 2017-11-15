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
word_list,w2i,i2w, w2v = data_import.read_and_create_dictionaries(corpus,vector_file)

if len(w2i) != len(i2w):
    raise NotImplementedError ('Length of w2i and i2w are not the same!!!')

N = 3
dim = 50
input_size = (N - 1) * dim
hidden_size = dim * 10
num_classes = len(w2i)
print ('There are',num_classes,'classes')
print ('There are',len(w2v),'vectors')
print ('The wordlist contains',len(word_list),'words')
num_epochs = 10
learning_rate = 0.001

ngram = data_import.generate_context(N, word_list) # Create ngrams
print ('Created ngrams!')

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.l1(x)
        x = F.tanh(x)
        x = self.l2(x)
        x = F.softmax(x)
        return x


mlp = Net(input_size, hidden_size, num_classes)

criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    random.shuffle(ngram)
    total_loss = torch.Tensor([0])
    for context, target in ngram:
        # Define input vector
        input_vector = None
        for word in context:
            try:
                input_vector = np.r_[input_vector, w2v[word]]
            except:
                input_vector = w2v[word]

        if len(input_vector) != (N - 1) * dim:
            # Check if input_vector if of the correct dimension
            print ('Context is',context)
            raise NotImplementedError('Length is not 100, there is probably an unknown word')

        np.reshape(input_vector, (len(input_vector), 1))
        input_vector = autograd.Variable(torch.FloatTensor(input_vector))
        # Done defining input vector

        mlp.zero_grad()
        optimizer.zero_grad()
        output = mlp(input_vector).unsqueeze(0)

        # Calculate the loss
        loss = criterion(output, autograd.Variable(torch.LongTensor([w2i[target]-1])))
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    print('After epoch',epoch,'the total loss is',total_loss)