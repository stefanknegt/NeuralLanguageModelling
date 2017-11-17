import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import random

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
        out = F.log_softmax(out)
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

            # if iter_count % 100 == 0:
            # print ('in epoch',epoch,'it is now at iter',100*iter_count/len(ngram),'%')
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