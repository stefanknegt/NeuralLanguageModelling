import numpy as np
import read_data as data_import
import model_batch as model
import torch
from torch import autograd
import time

train_file = "train.txt"
valid_file = "valid.txt"
test_file = "test.txt"

start_time = time.time()

N = 3
BPTT = 35
batch_size = 100
word_list,w2i,i2w = data_import.read_text(train_file,N,BPTT)
dim = 100
input_size = (N - 1) * dim
hidden_size = 500
num_classes = len(w2i)
print ('There are',num_classes,'classes')
num_epochs = 50

ngram = data_import.generate_context(N, word_list) # Create ngrams
print ('Created ngrams! There are',len(ngram))
name = 'Bengio_'+str(dim)+'_'+str(batch_size)+'_'+str(N)+'.pt'
#try:
#    trained_model = torch.load(name)
#except:
mlp = model.Net(dim, input_size, hidden_size, num_classes)
trained_model = model.train(N,num_epochs,ngram,w2i,mlp,batch_size)
torch.save(trained_model,name)

#start_sentence = [i2w[w2i['<s>']] for i in range(N-1)]
#sentence = model.next_word(N,start_sentence,w2i,i2w,trained_model)
#while sentence[-1] != '</s>':
#    sentence = model.next_word(N,sentence,w2i,i2w,trained_model)
#print (sentence)

print ('training one epochlasted',time.time()-start_time)

print ('We just trainend a',N,'gram')
print ('This model trained',num_epochs,'epochs')
print ('The embedding size was',dim)
print ('The batch size was',batch_size)
print ('There were',hidden_size,'hidden neurons')

perplexity_train,_ = model.calculate_perplexity(N,word_list,w2i,trained_model)
print ("The perplexity of the training set is",perplexity_train)

word_list_valid,_,_ = data_import.read_text(valid_file,N,BPTT)
perplexity_valid,_ = model.calculate_perplexity(N,word_list_valid,w2i,trained_model)
print ("The perplexity of the validation set is",perplexity_valid)

word_list_test,_,_ = data_import.read_text(test_file,N,BPTT)
perplexity_test,_ = model.calculate_perplexity(N,word_list_test,w2i,trained_model)
print("The perplexity of the test set is",perplexity_test)

print('It took',time.time()-start_time,'seconds')
