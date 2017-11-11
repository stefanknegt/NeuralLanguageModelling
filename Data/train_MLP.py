import numpy as np
import generate_test_vectors as gen_vec
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
<<<<<<< HEAD
import torch.nn.functional as F
from torch import autograd
=======
>>>>>>> master
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
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        x = self.l1(x)
        x = F.tanh(x)
        x = self.l2(x)
        x = F.softmax(x)
        return x

mlp = Net(input_size,hidden_size,num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)


print ('word_list length is',len(word_list))
print ('len of w2i is',len(w2i))

for i in range(N-1,len(word_list)):

    # Define input vector x
    x = None


    for j in range(0,N-1):
        try:
            x = np.r_[x,w2v[word_list[j]]]
        except:
            x = w2v[word_list[j-N+1]]

    # Done with defining  x

    np.reshape(x,(len(x),1))
    x = Variable(torch.FloatTensor(x))
    optimizer.zero_grad()
    outputs = mlp(x)

    index = w2i[word_list[i]]
    one_hot_target = torch.zeros(num_classes,1)

    #print ('index is',index)
    one_hot_target[index] = 1
    #print (one_hot_target)

    # loss = criterion(outputs,target)
    #loss.backward()
    #optimizer.step()


