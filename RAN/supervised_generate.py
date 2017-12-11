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
import data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from collections import Counter


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
#torch.manual_seed(args.seed)
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

with open('data/penn/test.txt') as f:
    sentences = f.readlines()
sentences = [x.strip() for x in sentences]
results = {}
results_second = {}
word_dependency = []
second_word_dependency = []
distances = []
all_words = []
print('There are ',len(sentences),' sentences...')
counter = 0
for sentence in sentences:
    sentence_list = sentence.split(' ')
    all_words += sentence_list
    #print(sentence_list)
    idx_list = [corpus.dictionary.word2idx[s] for s in sentence_list]

    settings.init()
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
    input.data.fill_(idx_list[0])
    for i in range(len(sentence_list)):
        output, hidden = model.forward(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        if i < len(sentence_list)-1:
            word_idx = idx_list[i+1]
            input.data.fill_(word_idx)

    assert(len(settings.iList) == len(settings.fList))
    word_list = np.zeros((len(settings.iList),len(settings.iList)))
    num_words = len(settings.iList)
    for word in range(0,num_words):
        for depth in range(0,num_words):
            if (word>depth):
                forget = settings.fList[depth+1]
                for k in range(depth+2,word+1):
                    forget = forget*settings.fList[k]
                answer = (torch.mm(settings.iList[depth],forget.transpose(0,1)))
                answer = answer.data.numpy()
                #print(depth,word,answer)
                word_list[depth][word] = answer
    mean_distance = np.zeros(num_words)
    #print(word_list)

    for i in range(1,num_words):
        average_index = 0
        for j in range(0,num_words):
            average_index += word_list[j][i] * j
        mean_distance[i] = i - average_index / np.sum(word_list[:,i])

    max_dependency = np.argmax(word_list, axis=0)
    word_list_second = np.array(word_list)
    for i in range(len(max_dependency)):
        word_list_second[max_dependency[i],i] = 0
    second_dependency = np.argmax(word_list_second, axis=0)
    distance =  np.arange(0,num_words) - max_dependency
    distances += distance.tolist()
    second_distance = np.arange(0,num_words) - second_dependency
    word_dependency += [sentence_list[i] for i in max_dependency]
    second_word_dependency += [sentence_list[i] for i in second_dependency]
    results[sentence] = distance
    results_second[sentence] = second_distance
    counter += 1
    if counter % 100 == 0:
        print('We are at sentence ',counter)
    """
    if num_words == 10:
        plt.imshow(word_list, cmap=CM.Blues, interpolation='nearest')
        x = np.array([0,1,2,3,4,5,6,7,8,9])
        my_xticks = [sentence_list[0], sentence_list[1], sentence_list[2], sentence_list[3],sentence_list[4],sentence_list[5],sentence_list[6],sentence_list[7],sentence_list[8],sentence_list[9]]
        plt.xticks(x, my_xticks)
        plt.yticks(x, my_xticks)
        plt.tight_layout()
        plt.title("Weight dependency heatmap")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    """

common_relative_dependency = Counter(word_dependency)
for key in common_relative_dependency:
    if all_words.count(key) > 10:
        common_relative_dependency[key] /= all_words.count(key)
    else:
        common_relative_dependency[key] = 0
most_common_relative = common_relative_dependency.most_common(10)
items = []
values = []
for i in most_common_relative:
    items.append(i[0])
    values.append(i[1])

indexes = np.arange(len(items))
width = 1
plt.bar(indexes, values, width)
plt.xticks(indexes, items)
plt.xlabel('10 words which are relatively most often max dependent')
plt.ylabel('Relative occurences')
plt.xticks(rotation=30)
plt.ylim([2,3])
plt.show()

averages = {}
for key in results:
    avg_dependency = sum(results[key])/len(results[key])
    averages[key]=avg_dependency
maximums = {}
for key in results:
    max_dependency = max(results[key])
    maximums[key]=max_dependency

#top_5 = sorted(maximums, key=maximums.get, reverse=True)[:5]
#results_top_5 = [results[i] for i in top_5]
#second_top_5 = [results_second[i] for i in top_5]
#print(top_5)
#print(results_top_5)
#print(second_top_5)
print('Here are the 100 sentences with max average dependency')
top_5 = sorted(averages, key=averages.get, reverse=True)[:300]
results_top_5 = [results[i] for i in top_5]
second_top_5 = [results_second[i] for i in top_5]
for i in range(len(top_5)):
    word_list = top_5[i].split(' ')
    if word_list.count('<unk>')==0 and word_list.count('N')==0:
        print(top_5[i], results_top_5[i], second_top_5[i])

most_common_dependencies = Counter(word_dependency).most_common(10)
items = []
values = []
for i in most_common_dependencies:
    items.append(i[0])
    values.append(i[1])

indexes = np.arange(len(items))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes, items)
plt.xlabel('10 words which were most dependent for prediction')
plt.ylabel('Occurences')
plt.xticks(rotation=30)
plt.ylim([400,1300])
plt.show()

x_axis = np.arange(max(distances)+1)
y_axis = np.zeros(len(x_axis))
for x in x_axis:
    y_axis[x] = distances.count(x)
plt.bar(x_axis, y_axis)
plt.xlabel('Distance of maximum dependent word')
plt.ylabel('Occurences')
plt.show()

x_axis = np.arange(5)
y_axis = np.zeros(len(x_axis))
for x in x_axis:
    y_axis[x] = distances.count(x)
plt.bar(x_axis, y_axis)
plt.xlabel('Distance of maximum dependent word (<5)')
plt.ylabel('Occurences')
plt.show()
