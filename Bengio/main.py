import numpy as np
import read_data as data_import
import model
import torch
from torch import autograd

train_file = "train-test.txt"
#valid_file = "valid.txt"
#test_file = "test.txt"

N = 5
word_list,w2i,i2w = data_import.read_text(train_file, N)
dim = 50
input_size = (N - 1) * dim
hidden_size = dim * 10
num_classes = len(w2i)
print ('There are',num_classes,'classes')
num_epochs = 50

ngram = data_import.generate_context(N, word_list) # Create ngrams
print ('Created ngrams!')

mlp = model.Net(dim, input_size, hidden_size, num_classes)
trained_model = model.train(N,num_epochs,ngram,w2i,mlp)

#start_sentence = [i2w[w2i['<s>']] for i in range(N-1)]
#
#sentence = model.next_word(N,start_sentence,w2i,i2w,trained_model)
#
#while sentence[-1] != '</s>':
#    sentence = model.next_word(N,sentence,w2i,i2w,trained_model)
#
#print (sentence)

perplexity_train,_ = model.calculate_perplexity(N,word_list,w2i,trained_model)
print ("The perplexity of the training set is",perplexity_train)

# word_list_valid,_,_ = data_import.read_text(valid_file,N)
# word_list_test,_,_ = data_import.read_text(test_file,N)
#
# perplexity_valid,_ = model.calculate_perplexity(N,word_list_valid,w2i,trained_model)
# print ("The perplexity of the validation set is",perplexity_valid)
#
# perplexity_test,_ = model.calculate_perplexity(N,word_list_test,w2i,trained_model)
# print ("The perplexity of the test set is",perplexity_test)

