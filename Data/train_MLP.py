import numpy as np
import generate_test_vectors as gen_vec
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

outname = "necessary-vectors.txt"
w2v,w2i,i2w,word_list = gen_vec.create_dictionaries(outname)

N = 4
dim = 50
input_size = (N-1) * dim
hidden_size = dim * 10
num_classes = len(i2w)
num_epochs = 5
learning_rate = 0.001

class Net(nn.Module):
    def __init__(self,input_size,hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size,num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

mlp = Net(input_size,hidden_size,num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

for i in range(N-1,len(word_list)):

    # Define input vector x
    x = None

    for j in range(0,N-1):
        try:
            x = np.r_[x,w2v[word_list[j]]]
        except:
            x = w2v[word_list[j-N+1]]

    np.reshape(x,(len(x),1))
    x = torch.FloatTensor(x)
    optimizer.zero_grad()
    outputs = mlp(x)
    print (outputs)
    #target = w2v[word_list[i]]
