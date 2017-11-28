import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import random
import read_data as data_import
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, dim, input_size,hidden_size, num_classes):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(num_classes,dim)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.bias2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        size = len(x)
        embeds = self.embeddings(x).view((size,-1))
        out = F.relu(self.l1(embeds))
        out = self.l2(out)
        out = F.log_softmax(out)
        return out

def train(N,num_epochs,ngram,w2i,mlp,batch_size):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001)

    print("Putting in large input matrix")
    input_matrix = np.zeros((len(ngram),N-1))
    target_matrix = np.zeros(len(ngram))
    counter = 0

    for context, target in ngram:
        input_matrix[counter,:] = np.asarray([w2i[w] for w in context]).reshape(1,len(context))
        target_matrix[counter] = np.asarray(w2i[target])

        if counter % 100000 == 0:
            print('Saved ',counter,' ngrams to the matrix')
        counter += 1

    print("Wooo finished")
    #torch_dataset = TensorDataset()
    for epoch in range(num_epochs):
        random.shuffle(ngram)
        total_loss = torch.Tensor([0])

        for b in range(0,input_matrix.shape[0], batch_size):
            input_batch = input_matrix[b:b+batch_size,:]
            target_batch = target_matrix[b:b+batch_size]
            #print(input_batch, target_batch)

            input_batch = autograd.Variable(torch.from_numpy(input_batch).long())
            target_batch = autograd.Variable(torch.from_numpy(target_batch).long())

            mlp.zero_grad()
            output = mlp(input_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.data

            if b % 1000 == 0:
                print ('in epoch',epoch,'it is now at batch',b/batch_size,' which is at ',(b*100)/len(ngram),'%')
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
            input_vector = autograd.Variable(torch.LongTensor(input_vector)).view((1,len(input_vector)))
            output = trained_model(input_vector)
            required_index = w2i[sentence[word]]

            sentence_prob += output[0][required_index].data[0]

            #print ('sentence_prob is',sentence_prob)
        test_set_prob += sentence_prob

    vocab = [item for sublist in word_list for item in sublist]
    number_of_words = 0

    for word in vocab:
        if word != '<s>' and word != '</s>':
            number_of_words += 1  # We do not count start and end symbols as words

    perplexity = np.exp ((-1.0 / float(number_of_words)) * test_set_prob )
    return perplexity, number_of_words
