import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import random
import read_data as data_import

class Net(nn.Module):
    def __init__(self, dim, input_size,hidden_size, num_classes):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(num_classes,dim)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.bias2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        embeds = self.embeddings(x).view((1,-1))
        out = F.relu(self.l1(embeds))
        out = self.l2(out)
        out = F.softmax(out)
        return out

def train(N,num_epochs,ngram,w2i,mlp):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)

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
                print('Context is', context)
                raise NotImplementedError('Length is not N-1, there is probably an unknown word')
            # Done defining input vector

            mlp.zero_grad()
            # optimizer.zero_grad()
            output = mlp(input_vector)

            # Calculate the loss and update the weights
            loss = criterion(output, autograd.Variable(torch.LongTensor([w2i[target]])))
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            iter_count += 1

            if iter_count % 1000 == 0:
                print ('in epoch',epoch,'it is now at iter',100.0*iter_count/len(ngram),'%')
        print('After epoch', epoch, 'the total loss is', total_loss)
    return mlp

def next_word(N,sentence,w2i,i2w,mlp):
    context = sentence[len(sentence)-N+1:]
    input_vector = [w2i[w] for w in context]
    input_vector = autograd.Variable(torch.LongTensor(input_vector))

    output = mlp(input_vector)
    values,index = torch.max(output,1) # returns maximum value and index of max
    new_word = i2w[index.data[0]]
    sentence.append(new_word)
    return sentence


def calculate_perplexity(N, word_list, w2i, trained_model):
    print("Starting to calculate the perplexity now!")
    sentence_list = word_list

    test_set_prob = 0.0
    for sentence in sentence_list:
        sentence_prob = 0.0
        for word in range(N-1, len(sentence)):
            context = [sentence[word - (N - 1) + i] for i in range(0, N - 1)]
            input_vector = [w2i[w] for w in context]
            input_vector = autograd.Variable(torch.LongTensor(input_vector))
            output = trained_model(input_vector)

            required_index = w2i[sentence[word]]
            sentence_prob += np.log(output[0][required_index].data[0])

            #print ('sentence_prob is',sentence_prob)
        test_set_prob += np.log2(np.exp(sentence_prob))

    vocab = [item for sublist in word_list for item in sublist]
    number_of_words = 0

    for word in vocab:
        if word != '<s>' and word != '</s>':
            number_of_words += 1  # We do not count start and end symbols as words

    perplexity = 2.0 ** ((-1.0 / float(number_of_words)) * test_set_prob )
    return perplexity, number_of_words