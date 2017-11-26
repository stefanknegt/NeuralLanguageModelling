###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
import settings
settings.init()
import data
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

written_words = []
with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden = model.forward(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        outf.write(word + ('\n' if i % 20 == 19 else ' '))
        written_words.append(word)
        #if word == '<eos>':
            #break
        if i == 100:
            break
        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))


assert(len(settings.iList) == len(settings.fList))
word_list = np.zeros((len(settings.iList),len(settings.iList)))
num_words = len(settings.iList)
for word in range(0,num_words):
    for depth in range(0,num_words):
        if (word>depth):
            forget = settings.fList[depth+1]
            for k in range(depth+2,word+1):
                #print(depth,word)
                forget = forget*settings.fList[k]
            answer = (torch.mm(settings.iList[depth],forget.transpose(0,1)))
            answer = answer.data.numpy()
            word_list[depth][word] = answer
print(word_list)
max_dependency = np.argmax(word_list, axis=0)
print(max_dependency)
word_dependency = [written_words[i] for i in max_dependency]
print(word_dependency)

distance = np.arange(0,len(max_dependency)) - max_dependency
plt.hist(distance)
plt.xlabel('Distance of highest dependent word')
plt.ylabel('Occurences')
plt.show()

mean_distance = np.zeros(num_words)
for i in range(1,num_words):
    average_index = 0
    for j in range(0,num_words):
        average_index += word_list[j][i] * j
    mean_distance[i] = i - average_index / np.sum(word_list[:,i])
plt.plot(mean_distance)
plt.xlabel('Word index generated')
plt.ylabel('Mean distance of dependent words')
plt.show()
